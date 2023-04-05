# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------
# Primary author: Hongyan Chang <hongyan.chang@intel.com>
# Co-authored-by: Anindya S. Paul <anindya.s.paul@intel.com>
# Co-authored-by: Brandon Edwards <brandon.edwards@intel.com>
# ------------------------------------------------------------

from copy import deepcopy
import functools
import torch.nn as nn
import torch.optim as optim
import torch
from torchio import DATA
import numpy as np
from openfl.experimental.interface import Aggregator, Collaborator
from local_brandon_copy_of_flspec import FLSpec
from openfl.experimental.runtime import LocalRuntime
from openfl.experimental.placement import aggregator, collaborator
import torchvision.transforms as transforms
import pickle
from pathlib import Path

from privacy_meter.model import PytorchModelTensor
from privacy_meter.dataset import Dataset
import copy
from auditor import (
    PopulationAuditor,
    plot_auc_history,
    plot_tpr_history,
    plot_roc_history,
    PM_report,
)

import time
import os
import argparse
import warnings

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5"
os.environ["CUDA_VISIBLE_DEVICES"]="0,3,4,5,6,7,8,9"

from GANDLF.parseConfig import parseConfig
from GANDLF.compute.generic import create_pytorch_objects
from GANDLF.compute.training_loop import train_network
from GANDLF.compute.forward_pass import validate_network
from GANDLF.utils import populate_header_in_parameters, parseTrainingCSV, populate_channel_keys_in_params, send_model_to_device, get_class_imbalance_weights
from GANDLF.models import get_model
from GANDLF.schedulers import get_scheduler
from GANDLF.optimizers import get_optimizer
from GANDLF.losses.segmentation import MCD

from GaNDLF_utils import GaNDLFLoaderWrapper, GaNDLFPyTorchModel
from get_loaders import get_loaders, get_single_loader, subject_to_feature, subject_to_label
warnings.filterwarnings("ignore")

# set the random seed for repeatable results
random_seed = 1234
torch.manual_seed(random_seed)

num_attempts = 5
num_subjects = 5

batch_size_train = 32
batch_size_test = 1000
learning_rate = 0.005
momentum = 0.9
log_interval = 10

# Brandon TODO: validate the use of 4 below
# FIXME: Validate the use of 4 below

loss_function = functools.partial(MCD, **{'num_class': 4, 'loss_type': 1})


def FedAvg(models, train_weights):  # NOQA: N802
    """
    Computes the non-weighted average of collaborator models

    Args:
        models: Python list of locally trained models by each collaborator
    """
    if len(models) != len(train_weights):
        raise ValueError(f"Asked to average {len(models)} models whith {len(train_weights)} weights.")
    new_model = models[0]
    if len(models) > 1:
        state_dicts = [model.state_dict() for model in models]
        state_dict = new_model.state_dict()
        for key in models[1].state_dict():
            state_dict[key] = torch.from_numpy(np.average(np.concatenate(
                [np.expand_dims(state[key], axis=0) for state in state_dicts], axis=0), axis=0, weights=train_weights))
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


def models_equal(model_1, model_2):
    equal = True
    for param_tensor in model_1.state_dict():
            if len(model_1.state_dict()[param_tensor].shape) == 0: 
               if model_1.state_dict()[param_tensor].item() != model_2.state_dict()[param_tensor].item():
                   equal = False 
            else:
                for tensor_1, tensor_2 in zip(
                    model_1.state_dict()[param_tensor],
                    model_2.state_dict()[param_tensor],
                ):
                    if (
                        torch.equal(tensor_1.to(device), tensor_2.to(device))
                        is not True
                    ):
                        equal = False
    return equal


def optimizer_to_device(optimizer, device):
    for param in optimizer.param_groups[0]['params']:
        param.data = param.data.to(device)
        if param.grad is not None:
            param.grad = param.grad.to(device)

# TODO: Make this work with GaNDLF models
def save_current_round_model_and_optimizer_for_next_round_testing(
    model, optimizer, collaborator_name, round_num
):
    """
    Save the model and optimizer state dictionary
    of a collaboartor ("collaborator_name")
    in a given round ("round_num") into a pickle file
    for later retieving and verifying its correctness.
    This provide the user the ability to verify the fields
    in the model and optimizer state dictionary and
    may provide confidence on the results of privacy auditing.
    Note: this functionality can be enabled through the command line
    argument by setting "--flow_internal_loop_test=True".

    Args:
        model: local collaborator model at the current round
        optimizer: local collaborator optimizer at the current round
        collaborator_name: name of the collaborator (Type:string)
        round_num: current round (Type:int)
    """
    model_config = {
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
    }
    with open(
        f"Collaborator_{collaborator_name}_model_config_roundnumber_{round_num}.pickle",
        "wb",
    ) as f:
        pickle.dump(model_config, f)


class FederatedFlow(FLSpec):
    def __init__(
        self,
        model,
        gandlf_config,
        device="cpu",
        total_rounds=10,
        top_model_accuracy=0,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.total_rounds = total_rounds
        self.top_model_accuracy = top_model_accuracy
        self.device = device
        self.round_num = 0  # starting round
        self.gandlf_config=gandlf_config
        self.gandlf_config["device"] = self.device
        self.verbose = verbose

        print(20 * "#")
        print(f"Round {self.round_num}...")
        print(20 * "#")

    @aggregator
    def start(self):
        self.start_time = time.time()
        print("Performing initialization for model")
        self.collaborators = self.runtime.collaborators
        self.private = 10

        self.next(
            self.aggregated_model_validation,
            foreach="collaborators",
            exclude=["private"],
        )

    # @collaborator  # Uncomment if you want ro run on CPU
    @collaborator(num_gpus=1)  # Assuming GPU(s) is available in the machine
    def aggregated_model_validation(self):

        # save off the global model before it gets trained (will audit it later)
        self.global_model = deepcopy(self.model)

        # Using collaborator private attributes to instantiate train, val, and test loaders
        self.train_loader, self.val_loader, self.gandlf_config = get_loaders(parameters=self.gandlf_config, 
                                                         prevent_shuffling=False, 
                                                         train_csv_path=self.target_train_path, 
                                                         val_csv_path=self.target_train_path)
     
        self.test_loader, _ = get_single_loader(parameters=self.gandlf_config, 
                                                     train=False, 
                                                     csv_path=self.target_test_path, 
                                                     prevent_shuffling=True) 
        
        self.train_weight = len(self.train_loader)
        self.val_weight = len(self.val_loader)
        self.test_weight = len(self.test_loader)

        print(f'Performing aggregated model validation for collaborator {self.input} on Device {self.device}')
        self.model = self.model.to(self.device)
        assert next(self.model.parameters()).device == self.device
        self.global_val_score = inference(network=self.model, 
                                              test_loader=self.val_loader, 
                                              scheduler=None, 
                                              round_num=self.round_num, 
                                              params=self.gandlf_config)
        self.global_test_score = inference(network=self.model, 
                                        test_loader=self.test_loader, 
                                        scheduler=None, 
                                        round_num=self.round_num, 
                                        params=self.gandlf_config)
        
        print(f'\n{self.input} global model validation score was: {self.global_val_score}')
        print(f'{self.input} global model test_score was: {self.global_test_score}\n')
        self.next(self.train)

    # @collaborator  # Uncomment if you want ro run on CPU
    @collaborator(num_gpus=1)  # Assuming GPU(s) is available on the machine
    def train(self):
        print(20 * "#")
        print(
            f"Performing model training for collaborator {self.input} in round {self.round_num}"
        )

        self.model.train()
        epochs = self.gandlf_config["num_epochs"]

        # temporarily utilizing an augmented gandlf config dictionary
        self.augmented_gandlf_config = deepcopy(self.gandlf_config)
        self.augmented_gandlf_config["model_parameters"] = model.parameters()
        optimizer = get_optimizer(self.augmented_gandlf_config)
        optimizer_to_device(optimizer=optimizer, device=self.device)
        if "optimizer_object" not in self.augmented_gandlf_config:
            self.augmented_gandlf_config["optimizer_object"] = optimizer
        if "scheduler" in self.augmented_gandlf_config:
            if not ("step_size" in self.augmented_gandlf_config["scheduler"]):
                self.augmented_gandlf_config["scheduler"]["step_size"] = (
                    self.augmented_gandlf_config["training_samples_size"] / self.augmented_gandlf_config["learning_rate"]
                )
            self.scheduler = get_scheduler(self.augmented_gandlf_config)
        else:
            self.scheduler = None
        

        for epoch in range(epochs):
            if epochs != 1.0:
                raise ValueError(f"Can remove this error, but wanted it to be clear only the last epoch is providing loss values.")
            print(f'Run {epoch} epoch of {self.round_num} round')
            epoch_train_loss, epoch_train_metric = train_network(model=self.model,
                                                                 train_dataloader=self.train_loader,
                                                                 optimizer=optimizer,
                                                                 params=self.augmented_gandlf_config)
        train_metric_dict = {'loss': epoch_train_loss}
        for k, v in epoch_train_metric.items():
            train_metric_dict[f'train_{k}'] = v
        self.local_train_dict = train_metric_dict
        print(f'{self.input} value of {self.local_train_dict}')

        delattr(self, 'train_loader')
    
        self.training_completed = True

        # sanity check that model and global model have diverted (rather than training on one reflecting in the other)
        if models_equal(model_1=self.model, model_2 = self.global_model):
            raise ValueError(f"Local update and global model are equal after training in round {self.round_num}, either they share memory or training was a no op!")
        
        self.next(self.local_model_validation)

    # @collaborator  # Uncomment if you want ro run on CPU
    @collaborator(num_gpus=1)  # Assuming GPU(s) is available in the machine
    def local_model_validation(self):
        print(
            (
                "Performing local model validation for collaborator: "
                f"{self.input} in round {self.round_num}"
            )
        )
        start_time = time.time()

        # Val dataset performance
        self.local_val_score = inference(network=self.model, 
                                                test_loader=self.val_loader, 
                                                scheduler=self.scheduler, 
                                                round_num=self.round_num, 
                                                params=self.augmented_gandlf_config)
        # Test dataset performance
        self.local_test_score = inference(network=self.model, 
                                                test_loader=self.test_loader, 
                                                scheduler=self.scheduler, 
                                                round_num=self.round_num, 
                                                params=self.augmented_gandlf_config)

        # remove val and test loader attributes
        delattr(self, 'val_loader')
        delattr(self, 'test_loader')
        delattr(self, 'augmented_gandlf_config')

        print(
            (
                "Doing local model validation for collaborator: "
                f"{self.input} validation: {self.local_val_score}"
                f"{self.input} test: {self.local_test_score}"
            )
        )
        print(f"local validation time cost {(time.time() - start_time)}")

        if (
            self.round_num == 0
            or self.round_num % self.local_pm_info.interval == 0
            or self.round_num == self.total_rounds
        ):
            self.next(self.audit)
        else:
            self.next(self.join, exclude=["training_completed"])

    # @collaborator  # Uncomment if you want ro run on CPU
    @collaborator(num_gpus=1)  # Assuming GPU(s) is available in the machine
    def audit(self):
        print(
            (
                "Performing local and global model auditing for collaborator: "
                f"{self.input} in round {self.round_num}"
            )
        )
        begin_time = time.time()

        # Note: The train boolean here is False for all since none of these are used for training

        x_train = GaNDLFLoaderWrapper(parameters=self.gandlf_config, 
                                      train=True, 
                                      type_restrictions='feature',
                                      subject_to_feature=functools.partial(subject_to_feature, **{'gandlf_config': self.gandlf_config}), 
                                      subject_to_label=functools.partial(subject_to_label, **{'gandlf_config': self.gandlf_config}),
                                      csv_path=self.PM_train_path, 
                                      verbose=self.verbose)

        y_train = GaNDLFLoaderWrapper(parameters=self.gandlf_config, 
                                      train=True, 
                                      type_restrictions='label',
                                      subject_to_feature=functools.partial(subject_to_feature, **{'gandlf_config': self.gandlf_config}), 
                                      subject_to_label=functools.partial(subject_to_label, **{'gandlf_config': self.gandlf_config}),
                                      csv_path=self.PM_train_path, 
                                      verbose=self.verbose)

        x_test = GaNDLFLoaderWrapper(parameters=self.gandlf_config, 
                                      train=True, 
                                      type_restrictions='feature',
                                      subject_to_feature=functools.partial(subject_to_feature, **{'gandlf_config': self.gandlf_config}), 
                                      subject_to_label=functools.partial(subject_to_label, **{'gandlf_config': self.gandlf_config}),
                                      csv_path=self.PM_test_path, 
                                      verbose=self.verbose)

        y_test = GaNDLFLoaderWrapper(parameters=self.gandlf_config, 
                                      train=True, 
                                      type_restrictions='label',
                                      subject_to_feature=functools.partial(subject_to_feature, **{'gandlf_config': self.gandlf_config}), 
                                      subject_to_label=functools.partial(subject_to_label, **{'gandlf_config': self.gandlf_config}),
                                      csv_path=self.PM_test_path, 
                                      verbose=self.verbose)

        x_pop = GaNDLFLoaderWrapper(parameters=self.gandlf_config, 
                                      train=True, 
                                      type_restrictions='feature',
                                      subject_to_feature=functools.partial(subject_to_feature, **{'gandlf_config': self.gandlf_config}), 
                                      subject_to_label=functools.partial(subject_to_label, **{'gandlf_config': self.gandlf_config}),
                                      csv_path=self.PM_pop_path, 
                                      verbose=self.verbose)

        y_pop = GaNDLFLoaderWrapper(parameters=self.gandlf_config, 
                                      train=True, 
                                      type_restrictions='label',
                                      subject_to_feature=functools.partial(subject_to_feature, **{'gandlf_config': self.gandlf_config}), 
                                      subject_to_label=functools.partial(subject_to_label, **{'gandlf_config': self.gandlf_config}),
                                      csv_path=self.PM_pop_path, 
                                      verbose=self.verbose)
  

        # The 'g' attribute defines the groups within which thresholds are computed independently
        # the value for 'g' could be set as a constant lists the same length as the'x' and 'y' 
        # attributes if all samples should fall in the same group 
        
        # grouping is not going to happen since we are segmenting
        train_dataset = {'x': x_train,
                        'y': y_train}
        test_dataset = {'x':  x_test,
                        'y': y_test}
        pop_dataset = {'x': x_pop, 
                    'y': y_pop}

        # now construct the dataset dict
        target_dataset = Dataset(data_dict={'train': train_dataset, 
                                            'test': test_dataset}, 
                                 default_input='x', 
                                default_output='y')
        pm_population_dataset = Dataset(data_dict={'train': pop_dataset}, 
                                        default_input='x', 
                                        default_output='y')

        """
        datasets = Dataset(data_dict=datasets,
                           default_input='x',
                           default_output='y')
        """

        """
        This is what was used here previously-----
        datasets = {
            "train": self.train_dataset,
            "test": self.test_dataset,
            "audit": self.population_dataset,
        }
        """
        
        start_time = time.time()
        # batch_size for the PytorchModelTensor indicates batch size for computing the signals.
        # for computing loss and logits, it can be large, e.g., 1000.
        # for computing the signal_norm, it should be around 25.
        # Otherwise, one may get OOM depending on the GPU memory.

        target_model = GaNDLFPyTorchModel(model_obj=copy.deepcopy(self.model).to(self.gandlf_config["device"]), 
                                          loss_fn=loss_function, 
                                          gandlf_config=self.gandlf_config)
        
        self.local_pm_info = PopulationAuditor(
            target_model, target_dataset, pm_population_dataset, self.local_pm_info
        )
        target_model.model_obj.to("cpu")
        self.local_pm_info.update_history("round", self.round_num)

        print(f"population attack for the local model uses {time.time() - start_time}")

        start_time = time.time()
        target_model = GaNDLFPyTorchModel(model_obj=self.global_model.to(self.gandlf_config["device"]), 
                                          loss_fn=loss_function, 
                                          gandlf_config=self.gandlf_config)
        self.global_pm_info = PopulationAuditor(
            target_model, target_dataset, pm_population_dataset, self.global_pm_info
        )
        self.global_pm_info.update_history("round", self.round_num)
        target_model.model_obj.to("cpu")
        print(f"population attack for the global model uses {time.time() - start_time}")

        start_time = time.time()

        history_dict = {
            "PM Result (Local)": self.local_pm_info,
            "PM Result (Global)": self.global_pm_info,
        }

        # # generate the plot for the privacy loss
        plot_tpr_history(history_dict, self.input, self.local_pm_info.fpr_tolerance)
        plot_auc_history(history_dict, self.input)
        plot_roc_history(history_dict, self.input)

        # save the privacy report
        saving_path = f"{self.local_pm_info.log_dir}/col_{self.input}_history_dict.pkl"
        Path(self.local_pm_info.log_dir).mkdir(parents=True, exist_ok=True)
        with open(saving_path, "wb") as handle:
            pickle.dump(history_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"auditing time: {time.time() - begin_time}")

        """
        TODO: Do we need to clean up anything here?
        # Clean up state before transitioning to collaborator
        delattr(self, "train_dataset")
        delattr(self, "train_loader")
        delattr(self, "test_dataset")
        delattr(self, "test_loader")
        delattr(self, "population_dataset")
        """
        self.next(self.join, exclude=["training_completed"])

    @aggregator
    def join(self, inputs):
        train_weights = np.array([input.train_weight for input in inputs])
        val_weights = np.array([input.val_weight for input in inputs])
        test_weights = np.array([input.test_weight for input in inputs])
        
        self.fed_local_loss = np.average([input.local_train_dict['loss'] for input in inputs], weights=train_weights)
        self.fed_local_val = np.average([input.local_val_score for input in inputs], weights=val_weights)
        self.fed_local_test = np.average([input.local_test_score for input in inputs], weights=test_weights)
        
        self.fed_global_val = np.average([input.global_val_score for input in inputs], weights=val_weights)
        self.fed_global_test = np.average([input.global_test_score for input in inputs], weights=test_weights)
        
        print(f'Average training loss = {self.fed_local_loss}')
        print(f'Global model validation DICE = {self.fed_global_val}')
        print(f'Global model test DICE = {self.fed_global_test}')
        print(f'Local model validation DICE = {self.fed_local_val}')
        print(f'Local model test DICE = {self.fed_local_test}')

        self.model = FedAvg([input.model.cpu() for input in inputs], train_weights)

        del inputs
        self.next(self.check_round_completion)

    @aggregator
    def check_round_completion(self):
        if self.round_num != self.total_rounds:
            if self.aggregated_valid_accuracy > self.top_model_accuracy:
                print(
                    (
                        "Validation accuracy improved to "
                        f"{self.aggregated_valid_accuracy} for round {self.round_num}"
                    )
                )
                self.top_model_accuracy = self.aggregated_valid_accuracy
            self.round_num += 1
            print()
            print(20 * "#")
            print(f"Round {self.round_num}...")
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
        print(20 * "#")
        print("All rounds completed successfully")
        print(20 * "#")
        print("This is the end of the flow")
        print(20 * "#")


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--csvdirpath',
        metavar="",
        type=str,
        help="Path to the collaborator partioned csv files")
    argparser.add_argument(
        '--config',
        metavar="",
        type=str,
        help="GaNDLF configuration file path")
    argparser.add_argument(
        "--signals",
        nargs="*",
        type=str,
        default=["loss"],
        help="Indicate which signal to use for membership inference attack",
    )
    argparser.add_argument(
        "--fpr_tolerance",
        nargs="*",
        type=float,
        default=[0.1, 0.5, 0.9],
        help="Indicate false positive tolerance rates in which users are interested",
    )
    argparser.add_argument(
        "--log_dir",
        type=str,
        default="test_debug",
        help="Indicate where to save the privacy loss profile and log files during the training",
    )
    argparser.add_argument(
        "--comm_round",
        type=int,
        default=30,
        help="Indicate the communication round of FL",
    )
    argparser.add_argument(
        "--auditing_interval", type=int, default=1, help="Indicate auditing interval"
    )
    argparser.add_argument(
        "--is_features",
        type=bool,
        default=True,
        help="Indicate whether to use the gradient norm with respect to the features as a signal",
    )
    argparser.add_argument(
        "--layer_number",
        type=int,
        default=10,
        help="Indicate whether layer to compute the gradient or gradient norm",
    )
    argparser.add_argument(
        "--flow_internal_loop_test",
        type=bool,
        default=False,
        help="Indicate enabling of internal loop testing of Federated Flow",
    )
    argparser.add_argument(
        "--init_model_path",
        type=str,
        default=None,
        help="The absolute path to the pretrained initial model.",
    )
    argparser.add_argument(
        "--verbose",
        action='store_true',
        help="The absolute path to the pretrained initial model.",
    )

    args = argparser.parse_args()

    # GaNDLF config
    gandlf_config_path = os.path.join(args.config)
    gandlf_config = parseConfig(gandlf_config_path)

    # Setup participants
    aggregator = Aggregator()
    aggregator.private_attributes = {}

    # Setup collaborators with private attributes
    collaborator_names = [str(n) for n in range(1,24)]
    collaborators = [Collaborator(name=name) for name in collaborator_names]
    
    if torch.cuda.is_available():
        device = torch.device(
            "cuda:0"
        )  # This will enable Ray library to reserve available GPU(s) for the task
    else:
        device = torch.device("cpu")

    for idx, collaborator in enumerate(collaborators):
        target_train_path = os.path.join(args.csvdirpath, 'train_val_test', collaborator.name, collaborator.name + "_train.csv")
        target_val_path = os.path.join(args.csvdirpath, 'train_val_test', collaborator.name, collaborator.name + "_val.csv")
        target_test_path = os.path.join(args.csvdirpath, 'train_val_test', collaborator.name, collaborator.name + "_test.csv")
        PM_train_path = os.path.join(args.csvdirpath, 'PM_train_test_pop', collaborator.name, "PM_tutorial_pm_train.csv")
        PM_test_path = os.path.join(args.csvdirpath, 'PM_train_test_pop', collaborator.name, "PM_tutorial_pm_test.csv")
        PM_pop_path = os.path.join(args.csvdirpath, 'PM_train_test_pop', collaborator.name, "PM_tutorial_pm_pop.csv")


        # initialize pm report to track the privacy loss during the training
        local_pm_info = PM_report(
            fpr_tolerance_list=args.fpr_tolerance,
            is_report_roc=True,
            level="local",
            signals=args.signals,
            log_dir=args.log_dir,
            interval=args.auditing_interval,
            other_info={
                "is_features": args.is_features,
                "layer_number": args.layer_number,
            },
        )
        global_pm_info = PM_report(
            fpr_tolerance_list=args.fpr_tolerance,
            is_report_roc=True,
            level="global",
            signals=args.signals,
            log_dir=args.log_dir,
            interval=args.auditing_interval,
            other_info={
                "is_features": args.is_features,
                "layer_number": args.layer_number,
            },
        )

        Path(local_pm_info.log_dir).mkdir(parents=True, exist_ok=True)
        Path(global_pm_info.log_dir).mkdir(parents=True, exist_ok=True)

        collaborator.private_attributes = {
            "local_pm_info": local_pm_info,
            "global_pm_info": global_pm_info,
            "target_train_path": target_train_path,
            "target_val_path": target_val_path,
            "target_test_path": target_test_path,
            "PM_train_path": PM_train_path, 
            "PM_test_path": PM_test_path,
            "PM_pop_path": PM_pop_path
        }
        
        
        
        
    # TODO: Below is to populate additional information into the gandlf_config. Using
    #       a function more targeted to that goal alone would be more optimal. We
    #       only  need to run this for the last collaborator's train and val paths
    # 

    _, _, gandlf_config = get_loaders(train_csv_path=target_train_path, 
                                      val_csv_path=target_val_path, 
                                      parameters=gandlf_config, 
                                      prevent_shuffling=False)
    



    # To activate the ray backend with parallel collaborator tasks run in their own process
    # and exclusive GPUs assigned to tasks, set LocalRuntime with backend='ray':
    local_runtime = LocalRuntime(aggregator=aggregator, collaborators=collaborators, backend='ray')

    print(f"Local runtime collaborators = {local_runtime.collaborators}")

    model = get_model(gandlf_config) 

    # If we have an initial model path, we will use it 
    if args.init_model_path:
        print(f"Loading a pretrained model as initial...")
        init_checkpoint = torch.load(args.init_model_path,map_location=torch.device('cpu'))
        model.load_state_dict(init_checkpoint['model_state_dict'])

    top_model_accuracy = 0

    """
    We will for now consider reinitialization of optimizers each round
    optimizers = {
        collaborator.name: default_optimizer(model, optimizer_type=args.optimizer_type)
        for collaborator in collaborators
    }
    """

    flflow = FederatedFlow(
        model=model,
        gandlf_config=gandlf_config, 
        device=device,
        total_rounds=args.comm_round,
        top_model_accuracy=top_model_accuracy,
        verbose=args.verbose
    )

    flflow.runtime = local_runtime
    flflow.run()
