from copy import deepcopy
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import numpy as np
import torchio
from tqdm import tqdm

import ray

from openfl.experimental.interface import Aggregator, Collaborator
from local_brandon_copy_of_flspec import FLSpec
# temporarily commenting below in favor of above for exploration
# from openfl.experimental.interface import FLSpec, Aggregator, Collaborator
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
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5"

import warnings
warnings.filterwarnings("ignore")

import pickle 

random_seed = 1234
torch.manual_seed(random_seed)


# Brandon trying to enable serialization of each data loader
class Brandon_loader(object):

    def __init__(self, info):
        self.info = info
        parameters, csv_path, train = self.info
        self.loader, parameters = get_single_loader(parameters=parameters, train=True, csv_path=csv_path)
        # parameters may have been modified above
        self.info = (parameters, csv_path, train)

    def __reduce__(self):
        unpack = Brandon_loader
        packaged_info = self.info
        return unpack, packaged_info

    def __iter__(self):
        return self.loader.__iter__()

    def __len__(self):
        return len(self.loader)

def gandlf_dict_to_feature(subject_dict, gandlf_config):
    return (torch.cat([subject_dict[key][DATA] for key in gandlf_config["channel_keys"]], 
                             dim=1).float().to(gandlf_config["device"]))
    
def gandlf_dict_to_label(subject_dict, gandlf_config):
    if len(subject_dict["value_0"].detach().cpu().numpy()) != 1:
        raise ValueError("Code expects batch size of one!")
    num_labels = len(gandlf_config['model']['class_list'])
    int_label = int(subject_dict["value_0"].detach().cpu().numpy().item())
    return np.eye(num_labels)[int_label]

def get_loaders(parameters, train_csv_path=None, val_csv_path=None):
    """
    This function creates the data loaders for each colaborator train, and test data.
    Args:
        parameters (dict): The parameters dictionary.
        train_csv_path (str): The path to the train CSV file.
        val_csv_path (str): The path to the test CSV file.
    Returns:
        train_loader (torch.utils.data.DataLoader): The training data loader.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
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

    return (train_loader, val_loader, parameters)


def get_single_loader(parameters, train, csv_path):
    """
    This function creates a data loader.
    Args:
        parameters (dict): The parameters dictionary.
        train (bool): Whether or not the loader will be used for training (augmentation, patching, ...)
        csv_path (str): The path to the CSV file.
    Returns:
        loader (torch.utils.data.DataLoader): A data loader for train or val or test.
    """

    # initialize loaders
    loader, headers = None, None

    if train:
        parameter_key = 'training_data'
    else:
        parameter_key = 'validation_data'

    # populate the data frames for the loader
    parameters[parameter_key], headers = parseTrainingCSV(csv_path, train=train)
    parameters = populate_header_in_parameters(parameters, headers)

    if train:
        loader = get_train_loader(parameters)
        parameters["training_samples_size"] = len(loader)
        # Calculate the weights here
        (
            parameters["weights"],
            parameters["class_weights"],
        ) = get_class_imbalance_weights(parameters["training_data"], parameters)

    else:
        # get the validation loader
        loader = get_validation_loader(parameters)

    return (loader, parameters)


def FedAvg(models):
    new_model = models[0]
    state_dicts = [model.state_dict() for model in models]
    state_dict = new_model.state_dict()
    for key in models[1].state_dict():
        state_dict[key] = torch.from_numpy(np.array(np.sum([state[key].detach().numpy() for state in state_dicts], axis=0, dtype=np.float) / len(models)))
    new_model.load_state_dict(state_dict)
    return new_model  

def inference(network, test_loader, scheduler, round_num, params):
    network.eval()
    epoch_valid_loss, epoch_valid_metric = validate_network(model=network,
                                                            valid_dataloader=test_loader,
                                                            scheduler=scheduler,
                                                            params=params,
                                                            epoch=round_num,
                                                            mode="validation")
    valid_metric_dict = {'loss': epoch_valid_loss}
    for k, v in epoch_valid_metric.items():
        valid_metric_dict[f'valid_{k}'] = v
    return valid_metric_dict

def optimizer_to_device(optimizer, device):
    for param in optimizer.param_groups[0]['params']:
        param.data = param.data.to(device)
        if param.grad is not None:
            param.grad = param.grad.to(device)




class FederatedFlow(FLSpec):

    def __init__(self, model, collaborator_names, device, total_rounds=10, top_model_accuracy=0, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.collaborator_names = collaborator_names
        self.total_rounds = total_rounds
        self.top_model_accuracy = top_model_accuracy
        self.device = device
        self.round_num = 0  

    # starting round
    @aggregator
    def start(self):
        print(f'Performing initialization for model')
        self.collaborators = self.runtime.collaborators
        self.private = 10

        # Brandon DEBUG
        print(f"Brandon DEBUG at start just before next is called, FLSpec._clones are: {FLSpec._clones}")

        self.next(self.aggregated_model_validation, foreach='collaborators', exclude=['private'])

    @collaborator(num_gpus=1)
    def aggregated_model_validation(self):
        print(f'Performing aggregated model validation for collaborator {self.input} on Device {self.device[self.input]}')
        
        self.train_loader = Brandon_loader((self.params, 
                                            self.train_csv_path, 
                                            True))
        self.params, _, _ = self.train_loader.info 
        self.val_loader = Brandon_loader((self.params, 
                                            self.val_csv_path, 
                                            False)) 
        self.params, _, _ = self.val_loader.info
        
        params = self.params 
        
        self.model = self.model.to(self.device[self.input])
        assert next(self.model.parameters()).device == self.device[self.input]

        # updating gandlf config
        params["model_parameters"] = model.parameters()
        self.optimizer = get_optimizer(params)
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
        
        self.agg_validation_score = inference(self.model, self.val_loader, self.scheduler, self.round_num, params)
        self.params = params

        print(f'{self.input} value of {self.agg_validation_score}')

        delattr(self, 'train_loader')
        delattr(self, 'val_loader')
        self.next(self.train)
    
    @collaborator(num_gpus=1)
    def train(self):
        print(f'Performing model training for collaborator {self.input} on Device {self.device[self.input]}')
        
        # Brandon DEBUG
        self.train_loader = Brandon_loader((self.params, 
                                            self.train_csv_path, 
                                            True))
        self.params, _, _ = self.train_loader.info 

        self.model.train()
        epochs = self.params["num_epochs"]
        for epoch in range(epochs):
            print(f'Run {epoch} epoch of {self.round_num} round')
            epoch_train_loss, epoch_train_metric = train_network(model=self.model,
                                                                 train_dataloader=self.train_loader,
                                                                 optimizer=self.optimizer,
                                                                 params=self.params)
        train_metric_dict = {'loss': epoch_train_loss}
        for k, v in epoch_train_metric.items():
            train_metric_dict[f'train_{k}'] = v
        self.local_train_score = train_metric_dict
        print(f'{self.input} value of {self.local_train_score}')
        self.training_completed = True

        delattr(self, 'train_loader')

        self.next(self.local_model_validation)

    @collaborator(num_gpus=1)
    def local_model_validation(self):

        # Brandon DEBUG
        self.val_loader = Brandon_loader((self.params, 
                                            self.val_csv_path, 
                                            False)) 
        self.params, _, _ = self.val_loader.info

        print(f'Performing local model validation for collaborator {self.input} on Device {self.device[self.input]}')

        self.local_validation_score = inference(self.model, self.val_loader, self.scheduler, self.round_num, self.params)
        
        delattr(self, 'val_loader')

        print(f'{self.input} value of {self.local_validation_score}')

        self.next(self.join, exclude=['training_completed'])

    @aggregator
    def join(self,inputs):
        self.average_train_loss = sum(input.local_train_score['loss'] for input in inputs)/len(inputs)
        self.average_aggregated_valid_loss = sum(input.agg_validation_score['loss'] for input in inputs)/len(inputs)
        self.average_local_valid_loss = sum(input.local_validation_score['loss'] for input in inputs)/len(inputs)
        self.aggregated_valid_accuracy = sum(input.agg_validation_score['valid_dice'] for input in inputs)/len(inputs)
        self.local_valid_accuracy = sum(input.local_validation_score['valid_dice'] for input in inputs)/len(inputs)
        self.local_train_accuracy = sum(input.local_train_score['train_dice'] for input in inputs)/len(inputs)
        print(f'Average training loss = {self.average_train_loss}')
        print(f'Average local training accuracy = {self.local_train_accuracy}')
        print(f'Average aggregared model validation loss = {self.average_aggregated_valid_loss}')
        print(f'Average aggregated model validation accuracy = {self.aggregated_valid_accuracy}')
        print(f'Average local model validation loss = {self.average_local_valid_loss}')
        print(f'Average local model validation accuracy = {self.local_valid_accuracy}')
        
        self.model = FedAvg([input.model.cpu() for input in inputs])

        self.next(self.check_round_completion)   # aggregated model

    @aggregator
    def check_round_completion(self):
        if self.round_num < self.total_rounds:
            if self.aggregated_valid_accuracy > self.top_model_accuracy:
                print(f'Accuracy improved to {self.aggregated_valid_accuracy} for round {self.round_num}')
                self.top_model_accuracy = self.aggregated_valid_accuracy
                self.round_num_top_accuracy = self.round_num

            self.round_num += 1
            print(20*"#")
            print(f'Round {self.round_num}...')
            print(20*"#")
            self.next(self.aggregated_model_validation, foreach='collaborators', exclude=['private'])
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        print(20*"#")
        print(f'All rounds completed successfully')
        print(20*"#")
        print(f'After round={self.total_rounds} aggegated model accuracy is {self.aggregated_valid_accuracy}')
        print(f'At round={self.round_num_top_accuracy} top model accuracy is {self.top_model_accuracy}')
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

    
    args = argparser.parse_args()

    from __main__ import FederatedFlow

    # GaNDLF config
    gandlf_config_path = os.path.join(args.config)
    gandlf_config = parseConfig(gandlf_config_path)

    if args.deviceid_multi:
        args.deviceid_multi = list(args.deviceid_multi)

    # Setup participants
    aggregator = Aggregator()
    aggregator.private_attributes = {}

    # Setup collaborators with private attributes
    # collaborator_names = [str(n) for n in range(1,4)]
    # Brandon DEBUG using below instead of above
    collaborator_names = [str(n) for n in range(1,2)]
    collaborators = [Collaborator(name=name) for name in collaborator_names]
    
    # Brandon DEBUG
    print(f"Brandon DEBUG: arsg: {args}")

    if args.gpu == 'single':
        # Brandon DEBUG
        print(f"Brandon DEBUG: args.gpu single and setting new value to device")
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

        # TODO: Below is to populate additional information into the gandlf_config. Using
        #       a function more targeted to that goal alone would be more optimal
        _, _, local_gandlf_config = get_loaders(train_csv_path=train_csv_path, 
                                                val_csv_path=val_csv_path, 
                                                parameters=gandlf_config)

        collaborator.private_attributes = {
                'train_csv_path': train_csv_path,
                'val_csv_path' : val_csv_path,
                'params'      : local_gandlf_config 
        }
        

    local_runtime = LocalRuntime(aggregator=aggregator, collaborators=collaborators, backend='ray')
    print(f'Local runtime collaborators = {local_runtime.collaborators}')

    # Now let's define some custom serialization methods

    def custom_brandon_loader_serializer(brandon_loader):
        return brandon_loader.info

    def custom_brandon_loader_deserializer(loader_info):
        return Brandon_loader(loader_info)

    # Register serializer and deserializer for class Brandon_loader:
    ray.util.register_serializer(Brandon_loader, 
                                 serializer=custom_brandon_loader_serializer, 
                                 deserializer=custom_brandon_loader_deserializer)

    """
    def custom_flflow_serializer(flflow_obj):
        if hasattr(flflow_obj, 'train_loader') and (flflow_obj.train_loader is not None):
            flflow_obj.train_loader = flflow_obj.train_loader.info
        else:
            flflow_obj.train_loader = None
        if hasattr(flflow_obj, 'val_loader') and (flflow_obj.val_loader is not None):
            flflow_obj.val_loader = flflow_obj.val_loader.info
        else:
            flflow_obj.val_loader = None
        return pickle.dumps(flflow_obj)

    def custom_flflow_deserializer(serialization):
        flflow_obj = pickle.loads(serialization)
        if hasattr(flflow_obj, 'train_loader') and (flflow_obj.train_loader is not None): 
            flflow_obj.train_loader = Brandon_loader(flflow_obj.train_loader)
        else:
            flflow_obj.train_loader = None
        if hasattr(flflow_obj, 'val_loader') and (flflow_obj.val_loader is not None):
            flflow_obj.val_loader = Brandon_loader(flflow_obj.val_loader)
        else:
            flflow_obj.val_loader = None
        return flflow_obj

    # Register serializer and deserializer for class FederatedFlow:
    ray.util.register_serializer(FederatedFlow, 
                                 serializer=custom_flflow_serializer, 
                                 deserializer=custom_flflow_deserializer)
    """
    
    # Here we use the last local config, there is no collaborator specific info used here by get_model however
    model = get_model(local_gandlf_config)
    top_model_accuracy = 0
    num_of_rounds = 10

    flflow = FederatedFlow(model=model, 
                           collaborator_names=None,
                           device=device,
                           total_rounds=num_of_rounds,
                           top_model_accuracy=top_model_accuracy)
    flflow.runtime = local_runtime
    # Brandon DEBUG
    deepcopy(flflow)
    print("BRANDON DEBUG, deepcopied succesfully before run")
    flflow.run()
