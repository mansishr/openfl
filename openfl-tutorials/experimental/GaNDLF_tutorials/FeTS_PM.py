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
from openfl.experimental.interface import FLSpec, Aggregator, Collaborator
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

from GANDLF.parseConfig import parseConfig
from GANDLF.compute.generic import create_pytorch_objects
from GANDLF.compute.training_loop import train_network
from GANDLF.compute.forward_pass import validate_network
from GANDLF.utils import populate_header_in_parameters, parseTrainingCSV, populate_channel_keys_in_params, send_model_to_device, get_class_imbalance_weights
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame
from GANDLF.models import get_model
from GANDLF.schedulers import get_scheduler
from GANDLF.optimizers import get_optimizer
from GANDLF.losses.segmentation import MCD

from GaNDLF_utils import get_loaders, GaNDLFLoaderWrapper, subject_to_feature, subject_to_label

warnings.filterwarnings("ignore")

# set the random seed for repeatable results
random_seed = 1234
torch.manual_seed(random_seed)

batch_size_train = 32
batch_size_test = 1000
learning_rate = 0.005
momentum = 0.9
log_interval = 10

# TODO: validate the use of 4 below
# FIXME: Validate the use of 4 below
loss_function = functools.partial(MCD, **{'num_classes': 4, 'loss_type': 1})




def FedAvg(models):  # NOQA: N802
    """
    Computes the non-weighted average of collaborator models

    Args:
        models: Python list of locally trained models by each collaborator
    """
    new_model = models[0]
    if len(models) > 1:
        state_dicts = [model.state_dict() for model in models]
        state_dict = new_model.state_dict()
        for key in models[1].state_dict():
            state_dict[key] = np.sum(
                [state[key] for state in state_dicts], axis=0
            ) / len(models)
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


# TODO: make this work with GaNDLF models
def load_previous_round_model_and_optimizer_and_perform_testing(
    model, global_model, optimizer, collaborator_name, round_num, device
):
    """
    Load pickle file to retrieve the model and optimizer state dictionary
    from the previous round for each collaborator
    and perform several validation routines with current
    round state dictionaries to test the flow loop.
    Note: this functionality can be enabled through the command line argument
    by setting "--flow_internal_loop_test=True".

    Args:
        model: local collaborator model at the current round
        global_model: Federated averaged model at the aggregator
        optimizer: local collaborator optimizer at the current round
        collaborator_name: name of the collaborator (Type:string)
        round_num: current round (Type:int)
        device: CUDA device id or "cpu"
    """
    print(f"Loading model and optimizer state dict for round {round_num-1}")
    model_prevround = Net()  # instanciate a new model
    model_prevround = model_prevround.to(device)
    optimizer_prevround = default_optimizer(model_prevround, optimizer_like=optimizer)
    if os.path.isfile(
        f"Collaborator_{collaborator_name}_model_config_roundnumber_{round_num-1}.pickle"
    ):
        with open(
            f"Collaborator_{collaborator_name}_model_config_roundnumber_{round_num-1}.pickle",
            "rb",
        ) as f:
            model_prevround_config = pickle.load(f)
            model_prevround.load_state_dict(model_prevround_config["model_state_dict"])
            optimizer_prevround.load_state_dict(
                model_prevround_config["optim_state_dict"]
            )

            for param_tensor in model.state_dict():
                for tensor_1, tensor_2 in zip(
                    model.state_dict()[param_tensor],
                    global_model.state_dict()[param_tensor],
                ):
                    if (
                        torch.equal(tensor_1.to(device), tensor_2.to(device))
                        is not True
                    ):
                        raise (
                            ValueError(
                                (
                                    "local and global model differ: "
                                    f"{collaborator_name} at round {round_num-1}."
                                )
                            )
                        )

                if isinstance(optimizer, optim.SGD):
                    if optimizer.state_dict()["state"] != {}:
                        for param_idx in optimizer.state_dict()["param_groups"][0][
                            "params"
                        ]:
                            for tensor_1, tensor_2 in zip(
                                optimizer.state_dict()["state"][param_idx][
                                    "momentum_buffer"
                                ],
                                optimizer_prevround.state_dict()["state"][param_idx][
                                    "momentum_buffer"
                                ],
                            ):
                                if (
                                    torch.equal(
                                        tensor_1.to(device), tensor_2.to(device)
                                    )
                                    is not True
                                ):
                                    raise (
                                        ValueError(
                                            (
                                                "Momentum buffer data differ: "
                                                f"{collaborator_name} at round {round_num-1}"
                                            )
                                        )
                                    )
                    else:
                        raise (ValueError("Current optimizer state is empty"))

                model_params = [
                    model.state_dict()[param_tensor]
                    for param_tensor in model.state_dict()
                ]
                for idx, param in enumerate(optimizer.param_groups[0]["params"]):
                    for tensor_1, tensor_2 in zip(param.data, model_params[idx]):
                        if (
                            torch.equal(tensor_1.to(device), tensor_2.to(device))
                            is not True
                        ):
                            raise (
                                ValueError(
                                    (
                                        "Model and optimizer do not point "
                                        "to the same params for collaborator: "
                                        f"{collaborator_name} at round {round_num-1}."
                                    )
                                )
                            )

    else:
        raise (ValueError("No such name of pickle file exists"))


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
        model_constructor,
        collaborator_names,
        optimizers,
        device="cpu",
        total_rounds=10,
        top_model_accuracy=0,
        flow_internal_loop_test=False,
        gandlf_config,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.global_model = model_constructor()
        self.optimizers = optimizers
        self.total_rounds = total_rounds
        self.top_model_accuracy = top_model_accuracy
        self.device = device
        self.flow_internal_loop_test = flow_internal_loop_test
        self.round_num = 0  # starting round
        self.gandlf_config=gandlf_config
        print(20 * "#")
        print(f"Round {self.round_num}...")
        print(20 * "#")

    @aggregator
    def start(self):
        self.start_time = time.time()
        print("Performing initialization for model")
        self.collaborators = self.runtime.collaborators
        self.private = 10

        train_loader_info = self.gandlf_config, \
                        True, \
                        self.target_train_path, \
                        'features_and_label', \
                        subject_to_feature, \
                        subject_to_label 
        self.train_loader_wrapper = GaNDLFLoaderWrapper(info=train_loader_info)
        self.gandlf_config = self.train_loader_wrapper.parameters

        val_loader_info = self.gandlf_config, \
                        False, \
                        self.target_val_path, \
                        'features_and_label', \
                        subject_to_feature, \
                        subject_to_label 
        self.val_loader_wrapper = GaNDLFLoaderWrapper(info=val_loader_info)
        self.gandlf_config = self.val_loader_wrapper.parameters

        test_loader_info = self.gandlf_config, \
                        False, \
                        self.target_test_path, \
                        'features_and_label', \
                        subject_to_feature, \
                        subject_to_label 
        self.test_loader_wrapper = GaNDLFLoaderWrapper(info=test_loader_info)
        self.gandlf_config = self.test_loader_wrapper.parameters

        self.next(
            self.aggregated_model_validation,
            foreach="collaborators",
            exclude=["private"],
        )

    # @collaborator  # Uncomment if you want ro run on CPU
    @collaborator(num_gpus=1)  # Assuming GPU(s) is available in the machine
    def aggregated_model_validation(self):
        print(f'Performing aggregated model validation for collaborator {self.input} on Device {self.device[self.input]}')
        params = self.gandlf_config   # load parameters from gandlf config
        self.model = self.model.to(self.device)
        assert next(self.model.parameters()).device == self.device

        # updating gandlf config
        params["model_parameters"] = model.parameters()
        self.optimizer = get_optimizer(params)
        params["optimizer_object"] = self.optimizer
        optimizer_to_device(optimizer=self.optimizer, device=self.device)
        if "scheduler" in params:
            if not ("step_size" in params["scheduler"]):
                params["scheduler"]["step_size"] = (
                    params["training_samples_size"] / params["learning_rate"]
                )
            self.scheduler = get_scheduler(params)
        else:
            self.scheduler = None
        params["device"] = self.device
        
        self.agg_validation_score = inference(self.model, self.val_loader_wrapper.base_loader, self.scheduler, self.round_num, params)
        self.agg_test_score = inference(self.model, self.test_loader_wrapper.base_loader, self.scheduler, self.round_num, params)
        self.params = params

        print(f'\n{self.input} validation score was: {self.agg_validation_score}')
        print(f'{self.input} test_score was: {self.agg_test_score}\n')
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
        for epoch in range(epochs):
            print(f'Run {epoch} epoch of {self.round_num} round')
            epoch_train_loss, epoch_train_metric = train_network(model=self.model,
                                                                 train_dataloader=self.train_loader_wrapper.base_loader,
                                                                 optimizer=self.optimizer,
                                                                 params=self.gandlf_config)
        train_metric_dict = {'loss': epoch_train_loss}
        for k, v in epoch_train_metric.items():
            train_metric_dict[f'train_{k}'] = v
        self.local_train_score = train_metric_dict
        print(f'{self.input} value of {self.local_train_score}')

        delattr(self, 'train_loader')
    
        self.training_completed = True
        
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
        print(self.device)
        start_time = time.time()

        print("Val dataset performance")
        self.local_validation_score = inference(
            self.model, self.val_loader, self.scheduler, self.round_num, self.device
        )
        print("Train dataset performance")
        self.local_train_score_train = inference(
            self.model, self.train_loader_wrapper.base_loader, self.scheduler, self.round_num, self.device
        )
        print("Test dataset performance")
        self.local_test_score_train = inference(
            self.model, self.test_loader_wrapper.base_loader, self.scheduler, self.round_num, self.device
        )

        # remove val and test loader attributes
        delattr(self, 'val_loader')
        delattr(self, 'test_loader')

        print(
            (
                "Doing local model validation for collaborator: "
                f"{self.input}: {self.local_validation_score}"
            )
        )
        print(f"local validation time cost {(time.time() - start_time)}")

        if (
            self.round_num == 0
            or self.round_num % self.local_pm_info.interval == 0
            or self.round_num == self.total_rounds
        ):
            print("Performing Auditing")
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

        loader_info_common = [self.gandlf_config, \
                        False, \
                        'features_and_labels', \
                        subject_to_feature, \
                        subject_to_label]
        
        PM_train_loader_info = tuple(loader_info_common[:2] + [self.PM_train_path] + loader_info_common[2:]) 
        self.PM_train_loader_wrapper = GaNDLFLoaderWrapper(info=PM_train_loader_info)
            
        
        PM_test_loader_info = tuple(loader_info_common[:2] + [self.PM_test_path] + loader_info_common[2:]) 
        self.PM_test_loader_wrapper = GaNDLFLoaderWrapper(info=PM_test_loader_info)

        PM_pop_loader_info = tuple(loader_info_common[:2] + [self.PM_pop_path] + loader_info_common[2:]) 
        self.PM_pop_loader_wrapper = GaNDLFLoaderWrapper(info=PM_pop_loader_info)

        # Note: The train boolean here is False for all since none of these are used for training
        
        x_loader_info_common = [self.gandlf_config, \
                                False, \
                                ('features'), None, \
                                subject_to_feature, \
                                subject_to_label]
        
        y_loader_info_common = [self.gandlf_config, \
                                False, \
                                ('labels'), None, \
                                subject_to_feature, \
                                subject_to_label]

       
        x_train_info = tuple(x_loader_info_common[:2] + [self.PM_train_path] + x_loader_info_common[2:])
        x_train = GaNDLFLoaderWrapper(info=x_train_info)

        y_train_info = tuple(y_loader_info_common[:2] + [self.PM_train_path] + y_loader_info_common[2:])
        y_train = GaNDLFLoaderWrapper(info=y_train_info)

        x_test_info = tuple(x_loader_info_common[:2] + [self.PM_test_path] + x_loader_info_common[2:])
        x_test = GaNDLFLoaderWrapper(info=x_test_info)

        y_test_info = tuple(y_loader_info_common[:2] + [self.PM_test_path] + y_loader_info_common[2:])
        y_test = GaNDLFLoaderWrapper(info=x_test_info)

        x_pop_info = tuple(x_loader_info_common[:2] + [self.PM_pop_path] + x_loader_info_common[2:])
        x_pop = GaNDLFLoaderWrapper(info=x_pop_info)

        y_pop_info = tuple(y_loader_info_common[:2] + [self.PM_pop_path] + y_loader_info_common[2:])
        y_pop = GaNDLFLoaderWrapper(info=y_pop_info)  
  

        # The 'g' attribute defines the groups within which thresholds are computed independently
        # the value for 'g' could be set as a constant lists the same length as the'x' and 'y' 
        # attributes if all samples should fall in the same group 
        
        # grouping is not going to happen since we are segmenting
        train_dataset = {'x': x_train,
                        'y': y_train, 
                        'g': y_train}
        test_dataset = {'x':  x_test,
                        'y': y_test, 
                        'g': y_test}
        pop_dataset = {'x': x_pop, 
                    'y': y_pop, 
                    'g': y_pop}

        # now construct the dataset dict
        datasets = {
                    'train': train_dataset,
                    'test': test_dataset,
                    'audit': pop_dataset
                    }

        datasets = Dataset(data_dict=datasets,
                           default_input='x',
                           default_output='y')

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

        #
        # 
        # 
        # Get the model and loss function from GaNDLF
        WORKING HERE


        target_model = PytorchModelTensor(
            copy.deepcopy(self.model), loss_function, self.device
        )
        self.local_pm_info = PopulationAuditor(
            target_model, datasets, self.local_pm_info
        )
        target_model.model_obj.to("cpu")
        self.local_pm_info.update_history("round", self.round_num)

        print(f"population attack for the local model uses {time.time() - start_time}")

        start_time = time.time()
        target_model = PytorchModelTensor(
            copy.deepcopy(self.global_model), loss_function, self.device
        )
        self.global_pm_info = PopulationAuditor(
            target_model, datasets, self.global_pm_info
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
        saving_path = f"{self.local_pm_info.log_dir}/{self.input}.pkl"
        Path(self.local_pm_info.log_dir).mkdir(parents=True, exist_ok=True)
        with open(saving_path, "wb") as handle:
            pickle.dump(history_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"auditing time: {time.time() - begin_time}")

        # Clean up state before transitioning to collaborator
        delattr(self, "train_dataset")
        delattr(self, "train_loader")
        delattr(self, "test_dataset")
        delattr(self, "test_loader")
        delattr(self, "population_dataset")
        self.next(self.join, exclude=["training_completed"])

    @aggregator
    def join(self, inputs):
        self.average_loss = sum(input.loss for input in inputs) / len(inputs)
        self.aggregated_model_accuracy = sum(
            input.agg_validation_score for input in inputs
        ) / len(inputs)
        self.local_model_accuracy = sum(
            input.local_validation_score for input in inputs
        ) / len(inputs)
        print(
            f"Average aggregated model validation values = {self.aggregated_model_accuracy}"
        )
        print(f"Average training loss = {self.average_loss}")
        print(f"Average local model validation values = {self.local_model_accuracy}")

        self.model = FedAvg([input.model.cpu() for input in inputs])
        self.global_model.load_state_dict(deepcopy(self.model.state_dict()))
        self.optimizers.update(
            {input.collaborator_name: input.optimizer for input in inputs}
        )

        del inputs
        self.next(self.check_round_completion)

    @aggregator
    def check_round_completion(self):
        if self.round_num != self.total_rounds:
            if self.aggregated_model_accuracy > self.top_model_accuracy:
                print(
                    (
                        "Accuracy improved to "
                        f"{self.aggregated_model_accuracy} for round {self.round_num}"
                    )
                )
                self.top_model_accuracy = self.aggregated_model_accuracy
            self.round_num += 1
            print(20 * "#")
            print(f"Round {self.round_num}...")
            print(20 * "#")
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
        default=["loss", "gradient_norm"],
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

    args = argparser.parse_args()

    # GaNDLF config
    gandlf_config_path = os.path.join(args.config)
    gandlf_config = parseConfig(gandlf_config_path)

    # Setup participants
    aggregator = Aggregator()
    aggregator.private_attributes = {}

    # Setup collaborators with private attributes
    collaborator_names = [str(n) for n in range(1,4)]
    collaborators = [Collaborator(name=name) for name in collaborator_names]
    
    if torch.cuda.is_available():
        device = torch.device(
            "cuda:0"
        )  # This will enable Ray library to reserve available GPU(s) for the task
    else:
        device = torch.device("cpu")

    for idx, collaborator in enumerate(collaborators):
        target_train_path = os.path.join(args.csvdirpath, ("_".join(["seg_test","train",collaborator.name])+".csv"))
        target_val_path = os.path.join(args.csvdirpath, ("_".join(["seg_test","val",collaborator.name])+".csv"))
        target_test_path = os.path.join(args.csvdirpath, ("_".join(["seg_test","test",collaborator.name])+".csv"))
        PM_train_path = os.path.join(args.csvdirpath, ("_".join(["seg_test","PM_train",collaborator.name])+".csv"))
        PM_test_path = os.path.join(args.csvdirpath, ("_".join(["seg_test","PM_test",collaborator.name])+".csv"))
        PM_pop_path = os.path.join(args.csvdirpath, ("_".join(["seg_test","PM_pop",collaborator.name])+".csv")) 


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
            "PM_train_path": PM_train_path, 
            "PM_test_path": PM_test_path,
            "PM_pop_path": PM_pop_path
        }
        
        
        
        
    # TODO: Below is to populate additional information into the gandlf_config. Using
    #       a function more targeted to that goal alone would be more optimal. We
    #       only  need to run this for the last collaborator's train and val paths
    # 
    """
    disabled for now ----   _, _, local_gandlf_config = get_loaders(train_csv_path=target_train_path, 
                                                val_csv_path=target_val_path, 
                                                parameters=gandlf_config)
    """

    X = np.concatenate([cifar_test.data, cifar_train.data])
    Y = np.concatenate([cifar_test.targets, cifar_train.targets]).tolist()

    train_dataset = deepcopy(cifar_train)
    train_dataset.data = X[:train_dataset_size]
    train_dataset.targets = Y[:train_dataset_size]

    test_dataset = deepcopy(cifar_test)
    test_dataset.data = X[train_dataset_size:train_dataset_size + test_dataset_size]
    test_dataset.targets = Y[
        train_dataset_size:train_dataset_size + test_dataset_size
    ]

    population_dataset = deepcopy(cifar_test)
    population_dataset.data = X[-audit_dataset_size:]
    population_dataset.targets = Y[-audit_dataset_size:]

    print(
        (
            f"Dataset info (total {N_total_samples}): "
            f"train - {len(train_dataset)}, "
            f"test - {len(test_dataset)}, "
            f"audit - {len(population_dataset)}"
        )
    )

    # partition the dataset for clients
    for idx, collab in enumerate(collaborators):

        # construct the training and test and population dataset
        local_train = deepcopy(train_dataset)
        local_test = deepcopy(test_dataset)
        local_population = deepcopy(population_dataset)

        local_train.data = train_dataset.data[idx::len(collaborators)]
        local_train.targets = train_dataset.targets[idx::len(collaborators)]

        local_test.data = test_dataset.data[idx::len(collaborators)]
        local_test.targets = test_dataset.targets[idx::len(collaborators)]

        local_population.data = population_dataset.data[idx::len(collaborators)]
        local_population.targets = population_dataset.targets[idx::len(collaborators)]

        



    # To activate the ray backend with parallel collaborator tasks run in their own process
    # and exclusive GPUs assigned to tasks, set LocalRuntime with backend='ray':
    local_runtime = LocalRuntime(aggregator=aggregator, collaborators=collaborators)

    print(f"Local runtime collaborators = {local_runtime.collaborators}")

    # change to the internal flow loop
    model = Net()
    top_model_accuracy = 0
    optimizers = {
        collaborator.name: default_optimizer(model, optimizer_type=args.optimizer_type)
        for collaborator in collaborators
    }
    flflow = FederatedFlow(
        model,
        optimizers,
        device,
        args.comm_round,
        top_model_accuracy,
        args.flow_internal_loop_test,
        gandlf_config
    )

    flflow.runtime = local_runtime
    flflow.run()
