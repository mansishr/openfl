import os
import yaml
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import numpy as np
import torchio
from tqdm import tqdm
import ray

from openfl.experimental.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.runtime import LocalRuntime
from openfl.experimental.placement import aggregator, collaborator

import argparse
from torchio import DATA
from torch.utils.data import TensorDataset, DataLoader

from GANDLF.parseConfig import parseConfig
from GANDLF.data import (
    get_train_loader,
    get_validation_loader,
)
from GANDLF.compute.generic import create_pytorch_objects
from GANDLF.compute.training_loop import train_network
from GANDLF.compute.forward_pass import validate_network
from GANDLF.utils import populate_header_in_parameters, parseTrainingCSV, populate_channel_keys_in_params, send_model_to_device, get_class_imbalance_weights
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.models import get_model
from GANDLF.schedulers import get_scheduler
from GANDLF.optimizers import get_optimizer

from torch.distributions.normal import Normal
from opacus.accountants.rdp import RDPAccountant
from opacus.data_loader import DPDataLoader
from torch.utils.data import DataLoader
from clip_optimizer import ClipOptimizer

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5"

random_seed = 1234
torch.manual_seed(random_seed)

# def gandlf_dict_to_feature(subject_dict, gandlf_config):
#     return (torch.cat([subject_dict[key][DATA] for key in gandlf_config["channel_keys"]], 
#                              dim=1).float().to(gandlf_config["device"]))
    
# def gandlf_dict_to_label(subject_dict, gandlf_config):
#     if len(subject_dict["value_0"].detach().cpu().numpy()) != 1:
#         raise ValueError("Code expects batch size of one!")
#     num_labels = len(gandlf_config['model']['class_list'])
#     int_label = int(subject_dict["value_0"].detach().cpu().numpy().item())
#     return np.eye(num_labels)[int_label]


def get_loaders(parameters, train_csv_path=None, val_csv_path=None):
    """
    This function creates the data loaders for each colaborator train, and test data.
    Args:
        parameters (dict): The parameters dictionary.
        train_csv_path (str): The path to the train CSV file.
        test_csv_path (str): The path to the test CSV file.
    Returns:
        train_loader (torch.utils.data.DataLoader): The training data loader.
        test_loader (torch.utils.data.DataLoader): The testing data loader.
    """

    # initialize loaders
    train_loader, val_loader = None, None
    headers_train, headers_val = None, None

    # populate the data frames for the train loader
    if train_csv_path is not None:
        parameters["training_data"], headers_train = parseTrainingCSV(
            train_csv_path, train=True
        )
        parameters = populate_header_in_parameters(parameters, headers_train)

    # get the train loader
        train_loader = get_train_loader(parameters)
        parameters["training_samples_size"] = len(train_loader)

        # Calculate the weights here
        (
            parameters["weights"],
            parameters["class_weights"],
        ) = get_class_imbalance_weights(parameters["training_data"], parameters)
    else:
        raise Exception("Train csv data is required")

    # populate the data frames for the test loader
    if val_csv_path is not None:
        parameters["validation_data"], headers_val = parseTrainingCSV(
            val_csv_path, train=False
        )

        if headers_train is None:
            parameters = populate_header_in_parameters(
                    parameters, headers_val
                )

        # get the validation loader
        val_loader = get_validation_loader(parameters)
    else:
        raise Exception("Validation csv data is required")

    return train_loader, val_loader, parameters

def FedAvg(models, previous_global_model=None, dp_params=None):  # NOQA: N802

    """
    Return a Federated average model based on Fedavg algorithm: H. B. Mcmahan,
    E. Moore, D. Ramage, S. Hampson, and B. A. Y.Arcas,
    “Communication-efficient learning of deep networks
    from decentralized data,” 2017.
    This tutorial utilizes non-weighted averaging of collaborator
    model updates regardless of whether DP config is used.
    Weighted FedAvg is currently not supported.

    Args:
        models: Python list of locally trained models by each collaborator
        at the current round
        previous_global_model: Federated averaged model from the previous round
        dp_params: Python dictionary for differential privacy
        specific hyperparameters as read from "test_config.yml"
    """
    if dp_params is not None and previous_global_model is not None:
        # Validate that in fact the local models clipped their updates
        non_delta_states = [model.state_dict() for model in models]
        previous_global_model_state = previous_global_model.cpu().state_dict()
        delta_states = []
        for non_delta_state in non_delta_states:
            delta_states.append(
                {
                    key: non_delta_state[key] - previous_global_model_state[key]
                    for key in non_delta_state
                }
            )
        # for idx, state in enumerate(delta_states):
        #     per_layer_norms = []
        #     for key, tensor in state.items():
        #         per_layer_norms.append(torch.norm(tensor.double(), dim=()))

        #     if torch.norm(torch.Tensor(per_layer_norms), dim=()) > dp_params["clip_norm"]:
        #         raise ValueError(
        #             f"The model with index {idx} had update whose "
        #             + "L2-norm was greater than clip norm."
        #             + "Correct the local periodic clipping."
        #         )
    new_model = models[0]
    state_dicts = [model.state_dict() for model in models]
    state_dict = models[0].state_dict()
    if len(state_dicts) > 1:
        for key in models[0].state_dict():
            if key.rsplit(".")[-1] != 'num_batches_tracked':
                state_dict[key] = np.sum(
                    [state[key] for state in state_dicts], axis=0
                ) / len(models)
    new_model.load_state_dict(state_dict)
    return new_model 

def inference(network, val_loader, scheduler, round, params):
    network.eval()
    epoch_valid_loss, epoch_valid_metric = validate_network(model=network,
                                                            valid_dataloader=val_loader,
                                                            scheduler=scheduler,
                                                            params=params,
                                                            epoch=round,
                                                            mode="validation")
    valid_metric_dict = {}
    valid_metric_dict = {'loss': epoch_valid_loss}
    for k, v in epoch_valid_metric.items():
        valid_metric_dict[f'valid_{k}'] = v
    return valid_metric_dict

def optimizer_to_device(optimizer, device):

    """
    Sending the "torch.optim.Optimizer" object into the specified device
    for model training and inference

    Args:
        optimizer: torch.optim.Optimizer from "default_optimizer" function
        device: CUDA device id or "cpu"
    """
    if optimizer.state_dict != {}:
        if isinstance(optimizer, optim.SGD):
            for param in optimizer.param_groups[0]["params"]:
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad = param.grad.to(device)
        elif isinstance(optimizer, optim.Adam):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
    else:
        raise (
            ValueError("Current optimizer state does not have dict keys: please verify")
        )


def clip_testing_on_optimizer_parameters(
    optimizer_before_step_params,
    optimizer_after_step_params,
    collaborator_name,
    round,
    device,
):
    """
    Test to check that optimizer parameters are clipped after performing
    optimizer step method.

    Args:
        optimizer_before_step_params: optimizer parameters before step
        optimizer_after_step_params: optimizer parameters after step
        collaborator_name: name of the collaborator (Type:string)
        round: current round (Type:int)
        device: CUDA device id or "cpu"
    """
    len_equal_tensor = 0
    for param_idx in range(len(optimizer_after_step_params)):
        for tensor_1, tensor_2 in zip(
            optimizer_before_step_params[param_idx],
            optimizer_after_step_params[param_idx],
        ):
            if torch.equal(tensor_1.to(device), tensor_2.to(device)) is True:
                len_equal_tensor += 1
    if len_equal_tensor == len(optimizer_after_step_params):
        raise (
            ValueError(
                "No clipping effect: Optimizer param data is the same "
                + "between before and after optimizer step for collaborator: "
                + f"{collaborator_name} at round {round}"
            )
        )


def validate_dp_params(dp_params):

    """
    The differential privacy block should have the exact keys as provided below.

    Args:
        dp_params: Python dictionary for differential privacy
        specific hyperparameters as read from "test_config.yml"
    """
    required_dp_keys = [
        "clip_norm",
        "noise_multiplier",
        "delta",
        "sample_rate",
        "clip_frequency",
    ]
    keys = dp_params.keys()
    excess = list(set(keys).difference(set(required_dp_keys)))
    deficit = list(set(required_dp_keys).difference(keys))

    if excess != []:
        print(
            f"\nCAUTION: The keys: {excess} where provided in the 'differential_privacy'"
            + "block of the flow config and are not being used.\n"
        )
    if deficit != []:
        raise ValueError(
            f"The 'differential_privacy' block is missing the required keys: {deficit}"
        )


def parse_config(config_path):

    """
    Parse "test_config.yml".

    Args:
        config_path: Path of "test_config.yml"
    """
    with open(config_path, "rb") as _file:
        config = yaml.safe_load(_file)
    return config


def add_noise_on_aggegated_parameters(collaborators, model, dp_params):

    """
    Adds noise on aggregated model parameters performed at the aggregator.

    Args:
        collaborators: Python list of collaborator name strings
        model: Federeated averaged model
        dp_params: Python dictionary for differential privacy
        specific hyperparameters as read from "test_config.yml"
    """
    state_dict = model.state_dict()
    normal_distribution = Normal(
        loc=0, scale=dp_params["noise_multiplier"] * dp_params["clip_norm"]
    )
    with torch.no_grad():
        for key in model.state_dict():
            if key.rsplit(".")[-1] != 'num_batches_tracked':
                noise_samples = normal_distribution.sample(
                    state_dict[key].shape
                ) / int(dp_params["sample_rate"] * float(len(collaborators)))
                state_dict[key].add_(noise_samples)
    model.load_state_dict(state_dict)
    return model


class FederatedFlow(FLSpec):

    def __init__(self, model, collaborator_names, device, dp_config, total_rounds=10, top_model_accuracy=0,
                    clip_test=False, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.previous_global_model = model
        self.collaborator_names = collaborator_names
        self.total_rounds = total_rounds
        self.top_model_accuracy = top_model_accuracy
        self.aggregated_valid_accuracy = 0
        self.device = device
        self.privacy_accountant = RDPAccountant()
        dp_config = parse_config(dp_config)
        self.clip_test = clip_test
        self.round = 0

        if "differential_privacy" not in dp_config:
            self.dp_params = None
        else:
            self.dp_params = dp_config["differential_privacy"]
            validate_dp_params(self.dp_params)                              

    @aggregator
    def start(self):
        print(f'Performing initialization for model')
        self.collaborators = self.runtime.collaborators
        self.private = 10

        if self.dp_params is None:
            self.round_collaborators = self.collaborators

        else:
            self.sample_rate = self.dp_params["sample_rate"]
            print("Collaborators are:", self.collaborators)
            global_data_loader = DataLoader(
                self.collaborators,
                batch_size=int(self.sample_rate * float(len(self.collaborators))),
            )
            dp_data_loader = DPDataLoader.from_data_loader(
                global_data_loader, distributed=False
            )
            collaborator_batch = []
            batch_sizes = []
            for cols in dp_data_loader:
                batch_sizes.append(len(cols))
                collaborator_batch.append(cols)
            print("Batches", collaborator_batch)
            self.round_collaborators = collaborator_batch[0]

        print("Collaborators sampled", self.round_collaborators)

        if len(self.round_collaborators) != 0:
            if not isinstance(self.round_collaborators[0], torch.Tensor):
                print(20 * "#")
                print(f"Round {self.round}...")
                print("Batch sizes sampled:", batch_sizes)
                print(
                    f"Collaborators patricipated in Round: {self.round}",
                    self.round_collaborators,
                )
                print(20 * "#")
                self.next(
                    self.aggregated_model_validation,
                    foreach="round_collaborators",
                    exclude=["private"],
                )
            else:
                print(f"No collaborator selected for training at Round: {self.round}")
                self.next(self.check_round_completion)
        else:
            print(f"No collaborator selected for training at Round: {self.round}")
            self.next(self.check_round_completion)

    @collaborator(num_gpus=1)
    def aggregated_model_validation(self):
        print(f'Performing aggregated model validation for collaborator {self.input} on Device {self.device[self.input]}')
        print("AMV ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        self.train_loader, self.val_loader, _ = get_loaders(parameters=self.params, 
                                                         train_csv_path=self.train_csv_path, 
                                                         val_csv_path=self.val_csv_path)
        
        self.model = self.model.to(self.device[self.input])
        self.previous_global_model = self.previous_global_model.to(self.device[self.input])
        # updating gandlf config
        params = deepcopy(self.params)
        params["model_parameters"] = self.model.parameters()
        # self.optimizer = get_optimizer(params)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        params["optimizer_object"] = self.optimizer
        optimizer_to_device(optimizer=self.optimizer, 
                    device=self.device[self.input])
        if "scheduler" in params:
            if not ("step_size" in params["scheduler"]):
                params["scheduler"]["step_size"] = (
                    params["training_samples_size"] / params["learning_rate"]
                )
            self.scheduler = get_scheduler(params)
        else:
            self.scheduler = None
        params["device"] = self.device[self.input]
        
        self.agg_validation_score = inference(self.model, self.val_loader, self.scheduler, self.round, params)
        self.augmented_params = params
        del params
        print(f'{self.input} value of {self.agg_validation_score}')
        self.next(self.train)
    
    @collaborator(num_gpus=1)
    def train(self):
        print(f'Performing model training for collaborator {self.input} on Device {self.device[self.input]}')
        print("TRAIN ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        if self.dp_params is not None:
            # base_optimizer = self.optimizer
            self.optimizer = ClipOptimizer(
                    base_optimizer=torch.optim.SGD(self.model.parameters(), lr=0.001),
                    device=self.device,
                    global_model_params=self.previous_global_model.parameters(),
                    clip_norm=self.dp_params["clip_norm"],
                    clip_freq=self.dp_params["clip_frequency"],
                )
        self.model = self.model.to(self.device[self.input])

        if self.dp_params and self.clip_test:
            optimizer_before_step_params = [
                param.data for param in self.optimizer.param_groups()[0]["params"]
            ]
        epochs = self.augmented_params["num_epochs"]
        model_before_train = deepcopy(self.model)

        for epoch in range(epochs):
            print(f'Run {epoch} epoch of {self.round} round')
            epoch_train_loss, epoch_train_metric = train_network(model=self.model,
                                                                 train_dataloader=self.train_loader,
                                                                 optimizer=self.optimizer,
                                                                 params=self.augmented_params)
        train_metric_dict = {'loss': epoch_train_loss}
        for k, v in epoch_train_metric.items():
            train_metric_dict[f'train_{k}'] = v
        self.local_train_score = train_metric_dict
        print(f'{self.input} value of {self.local_train_score}')
        if self.dp_params and self.clip_test:
            optimizer_after_step_params = [
                param.data for param in self.optimizer.param_groups()[0]["params"]
            ]
            clip_testing_on_optimizer_parameters(
                                optimizer_before_step_params,
                                optimizer_after_step_params,
                                self.collaborator_name,
                                self.round,
                                self.device,
                            )
        #Copy the 'num_batches_tracked' layer tensors back to the model
        for key,param in self.model.state_dict().items():
            if key.rsplit(".")[-1] == 'num_batchess_tracked':
                param.data = model_before_train.state_dict[key].data

        delattr(self, 'train_loader')
        self.training_completed = True

        self.next(self.local_model_validation)

    @collaborator(num_gpus=1)
    def local_model_validation(self):
        print(f'Performing local model validation for collaborator {self.input} on Device {self.device[self.input]}')
        print("LMV ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        self.local_validation_score = inference(self.model, self.val_loader, self.scheduler, self.round, self.augmented_params)
        
        print(f'{self.input} value of {self.local_validation_score}')
         # now deleting some attributes that resist pickling and we recover later
        delattr(self, 'val_loader')
        delattr(self, 'augmented_params')
        self.next(self.join, exclude=['training_completed'])

    ###########
    @aggregator
    def join(self, inputs):
        self.average_train_loss = sum(input.local_train_score['loss'] for input in inputs)/len(inputs)
        self.average_aggregated_valid_loss = sum(input.agg_validation_score['loss'] for input in inputs)/len(inputs)
        self.average_local_valid_loss = sum(input.local_validation_score['loss'] for input in inputs)/len(inputs)
        self.aggregated_valid_accuracy = sum(input.agg_validation_score['valid_dice'] for input in inputs)/len(inputs)
        self.local_valid_accuracy = sum(input.local_validation_score['valid_dice'] for input in inputs)/len(inputs)
        self.local_train_accuracy = sum(input.local_train_score['train_dice'] for input in inputs)/len(inputs)
        print(f'Average training loss = {self.average_train_loss}')
        print(f'Average local training accuracy = {self.local_train_accuracy}')
        print(f'Average aggregated model validation loss = {self.average_aggregated_valid_loss}')
        print(f'Average aggregated model validation accuracy = {self.aggregated_valid_accuracy}')
        print(f'Average local model validation loss = {self.average_local_valid_loss}')
        print(f'Average local model validation accuracy = {self.local_valid_accuracy}')

        if self.dp_params is not None:
            self.model = FedAvg(
                [input.model.cpu() for input in inputs],
                previous_global_model=self.previous_global_model,
                dp_params=self.dp_params,
            )
        else:
            self.model = FedAvg([input.model.cpu() for input in inputs])

        if self.dp_params is not None:
            self.model = add_noise_on_aggegated_parameters(
                self.collaborators, self.model, self.dp_params
            )
            self.privacy_accountant.step(
                noise_multiplier=self.dp_params["noise_multiplier"],
                sample_rate=self.sample_rate,
            )
            epsilon, best_alpha = self.privacy_accountant.get_privacy_spent(
                delta=self.dp_params["delta"]
            )
            print(20 * "#")
            print(
                f"\nCurrent privacy spent using delta={self.dp_params['delta']} "
                + f"is epsilon={epsilon} (best alpha was: {best_alpha})."
            )
            print(20 * "#")
        self.previous_global_model.load_state_dict(deepcopy(self.model.state_dict()))
        # self.optimizers.update(
        #     {input.collaborator_name: input.optimizer for input in inputs}
        # )

        del inputs
        self.next(self.check_round_completion)

    @aggregator
    def check_round_completion(self):
        if self.round != self.total_rounds:
            if self.aggregated_valid_accuracy > self.top_model_accuracy:
                print(
                    (
                        "Validation accuracy improved to "
                        f"{self.aggregated_valid_accuracy} for round {self.round}"
                    )
                )
                self.top_model_accuracy = self.aggregated_valid_accuracy
            self.round += 1
            print()
            print(20 * "#")
            print(f"Round {self.round}...")
            print(20 * "#")
            print()
            self.next(
                self.aggregated_model_validation,
                foreach="collaborators",
                exclude=["private"],
            )
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        print(20*"#")
        print(f'All rounds completed successfully')
        print(20*"#")
        print(f'This is the end of the flow')
        print(20*"#")

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--gpu',
        default="single",
        help='single(collaborators running on single GPU) or multi(each collaborator is assigned to a GPU) (default:"single")')
    argparser.add_argument(
        '--deviceid_single',
        default="0",
        help='Device id for single GPU (default:"0")')
    argparser.add_argument(
        '--deviceid_multi',
        default="0123",
        help='Device id for multiple GPU (default:"0123")')
    argparser.add_argument(
        '--config',
        metavar="",
        type=str,
        help="GaNDLF configuration file path")
    argparser.add_argument(
        '--csvdirpath',
        metavar="",
        type=str,
        help="Path to the collaborator partioned csv files")
    argparser.add_argument(
        "--dp_config",
        type=str,
        help="Path to the DP configuration file"
    )
    argparser.add_argument(
        "--clip_test",
        default=False,
        help="Indicate enabling of optimizer param testing before and after clip",
    )
    argparser.add_argument(
        "--init_model_path",
        help="Initial model",
    )

    
    args = argparser.parse_args()

    # GaNDLF config
    gandlf_config_path = os.path.join(args.config)
    gandlf_config = parseConfig(gandlf_config_path)

    if args.deviceid_multi:
        args.deviceid_multi = list(args.deviceid_multi)

    # Setup participants
    aggregator = Aggregator()
    aggregator.private_attributes = {}

    # Setup collaborators with private attributes
    collaborator_names = [str(x+1) for x in range(23)]
    # collaborator_names = [
    #     "1",
    #     "2",
    #     "3",
    # ]
    collaborators = [Collaborator(name=name) for name in collaborator_names]
    
    if torch.cuda.is_available():
        device = torch.device(
            "cuda:0"
        )  # This will enable Ray library to reserve available GPU(s) for the task
    else:
        device = torch.device("cpu")

    if args.gpu == 'single':
        if torch.cuda.is_available():
            device = {collaborators[i].name: torch.device(f'cuda:{args.deviceid_single}') for i in range(len(collaborator_names))}
        else:
           device = {collaborators[i].name: torch.device('cpu') for i in range(len(collaborator_names))}
    elif args.gpu == 'multi':
        if torch.cuda.is_available():
            device = {collaborators[i].name: torch.device(f'cuda:{args.deviceid_multi[i]}') for i in range(len(collaborator_names))}
        else:
           device = {collaborators[i].name: torch.device('cpu') for i in range(len(collaborator_names))}
    else:
        raise Exception('input should be single or multi')

    for idx, collaborator in enumerate(collaborators):
        train_csv_path = os.path.join(args.csvdirpath, ("_".join(["seg_test","train",collaborator.name])+".csv"))
        val_csv_path = os.path.join(args.csvdirpath, ("_".join(["seg_test","val",collaborator.name])+".csv"))
        _, _, local_gandlf_config = get_loaders(parameters=gandlf_config,
                                                train_csv_path=train_csv_path, 
                                                val_csv_path=val_csv_path)
        collaborator.private_attributes = {
                'train_csv_path': train_csv_path,
                'val_csv_path' : val_csv_path,
                'params'      : local_gandlf_config 
        }

    local_runtime = LocalRuntime(aggregator=aggregator, collaborators=collaborators, backend='ray')
    print(f'Local runtime collaborators = {local_runtime.collaborators}')
    
    model = get_model(local_gandlf_config)
    # /raid/edwardsb/projects/OpenFL/FeTS_PM_GDP_tutorials/models/resunet_pretrained.pth
    # model = torch.load('/raid/edwardsb/projects/OpenFL/FeTS_PM_GDP_tutorials/models/resunet_pretrained.pth')
    # model = get_model(gandlf_config)

    # If we have an initial model path, we will use it

    if args.init_model_path:
        print(f"Loading a pretrained model as initial...")
        init_checkpoint = torch.load(args.init_model_path,map_location=torch.device('cpu'))
        model.load_state_dict(init_checkpoint['model_state_dict'])

    top_model_accuracy = 0
    num_of_rounds = 10

    flflow = FederatedFlow(model=model,
                           collaborator_names=None,
                           device=device,
                           dp_config=args.dp_config,
                           total_rounds=num_of_rounds,
                           top_model_accuracy=top_model_accuracy,
                           clip_test=args.clip_test)
    flflow.runtime = local_runtime
    deepcopy(flflow)
    flflow.run()
