import numpy as np

import torch
from torchio import DATA

from GANDLF.data import (
    get_train_loader,
    get_validation_loader,
)
from GANDLF.models import global_models_dict
from GANDLF.utils import populate_header_in_parameters, parseTrainingCSV, populate_channel_keys_in_params, get_class_imbalance_weights

from privacy_meter.model import PytorchModel


def subject_to_feature(subject_dict, gandlf_config):
    features = torch.cat([subject_dict[key][DATA] for key in gandlf_config["channel_keys"]], 
                             dim=1).float().to(gandlf_config["device"])
    return features
    
def subject_to_label(subject_dict, gandlf_config):
    print(f"Shape of label is: ")
    return subject_dict["label"]["data"].float().to(gandlf_config["device"])
   
    
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

    train_loader, parameters = get_single_loader(parameters=parameters,
                                                    train=True,
                                                    csv_path=train_csv_path)
    val_loader, parameters = get_single_loader(parameters=parameters,
                                                train=False,
                                                csv_path=val_csv_path)

    return (train_loader, val_loader, parameters)


def get_model_info(parameters, loss_function):
    """
    This function gets the model class being used from the global models dict.
    Args:
        parameters (dict): The parameters dictionary.
        loss_function: The loss that will be used with the model, must allow for a parameter 'reduction'
                       for which the value indicates whether or not to return per sample loss or average over samples
    Returns:
        model_class (torch.nn.Module): The model to use for training.
        loss_function_w_reduction (): The loss function used with this model (averages loss over samples)
        loss_function_wo_reduction (): The loss function used with this model (does not average loss over samples)
    """

    # get the model class (here we use a vgg only global models dict since not using this script much, as it will 
    # be replaced when PM code is made more modular)
    model_class = global_models_dict[parameters["model"]["architecture"]]
    # get the loss function
    
    # TODO: support more losses using the global losses dict
    loss_function_w_reduction = loss_function()
    loss_function_wo_reduction = loss_function(reduction='none') # partial(CEL, **{'reduction': 'none'})
    
    return model_class, loss_function_w_reduction, loss_function_wo_reduction  




# Help GaNDLF loaders be treated like numpy arrays (slicing). Also, help deepcopy GaNDLF loaders (via __reduce__)
class GaNDLFLoaderWrapper(object):
    def __init__(self, 
                 parameters, 
                 train, 
                 csv_path, 
                 type_restrictions,
                 idx_restrictions, 
                 subject_to_feature, 
                 subject_to_label, 
                 base_loader=None):
        """
        TODO rewrite this documentation below-----
        restriction (tuple of: str, np.ndarray): First component can be 'feature', 'label', or
        'feature_and_label', array specifies which indices to allow during iteration. Note base loader
        must be deterministic. We insert a test for tjis determinism.
        """
        super().__init__()
            
        self.parameters = parameters
        self.train = train
        self.csv_path = csv_path
        self.type_restrictions = type_restrictions
        self.idx_restrictions = idx_restrictions
        self.subject_to_feature = subject_to_feature
        self.subject_to_label = subject_to_label
        self.type_restrictions = type_restrictions 
        self.idx_restrictions = idx_restrictions
        self.base_loader = base_loader
        
        if self.base_loader is None:
            self.base_loader, self.parameters = get_single_loader(parameters=self.parameters, 
                                                 train=self.train, 
                                                 csv_path=self.csv_path)
            
        # Try to catch the base loader breaking the assumption of determinism
        # (this check is specific to BraTS) - only checks 1st channel so possible to not catch
        chnl1_tensors = []
        subject_ids = []

        num_attempts = 3
        num_subjects = 5

        all_equal = True

        for a_idx, attempt in enumerate(range(num_attempts+1)):
            if a_idx == 0:
                for s_idx, subject in enumerate(self.base_loader):
                    if s_idx == num_subjects:
                        break
                    else:
                        chnl1_tensors.append(subject['1']['data'])
                        subject_ids.append(subject['subject_id'])
            else:
                print(f"Comparing one run of base loader with another...attempt={a_idx+1}")
                equal = True
                for s_idx, subject in enumerate(self.base_loader):
                    if s_idx == num_subjects:
                        break
                    else:
                        if not torch.equal(chnl1_tensors[s_idx], subject['1']['data']) or subject_ids[s_idx] != subject['subject_id']:
                            equal = False
                if not equal: 
                    all_equal = False
        if not all_equal:
            raise ValueError(f"Base GaNDLF loader is not deterministic and so loader wrapper will not work!")
        else:
            print(f"Base loader sent to GaNDLFLoaderWrapper appeared to be deterministic when tested against {num_attempts} attempts and checking only channel 1 of features.")        

        self.base_loader_length = len(self.base_loader)       
        # some parameter handling
        if self.idx_restrictions is None:
            self.idx_restrictions = np.arange(len(self.base_loader))

        # sanity check arguments
        if self.type_restrictions not in ["feature", "label", "feature_and_label"]:
            raise ValueError(
                "The first element of the restrictions tuple must be 'feature', 'label', or 'feature_and_label'."
            )
        if not isinstance(self.idx_restrictions, np.ndarray):
            raise ValueError(
                "The second element of the restrictions tuple must be a numpy array."
            )
        if len(self.idx_restrictions.shape) != 1:
            raise ValueError(
                "The second element of the restirctions tuple must have a shape of length one."
            )

        # initialize state
        self.base_iter = None
        self.base_iter_idx = None

    def iterate_to_next_restricted_idx_or_raise_stop(self):
        if self.base_iter_idx is None:
            raise ValueError("Iteration is progressing before base_iter_idx is not set.")
        if self.base_iter is None:
            raise ValueError("Iteration is progressing before base_iter is set.")
        iters_to_stop = self.base_loader_length - self.base_iter_idx + 1
        for i in range(iters_to_stop):
            deliver_result = self.base_iter_idx in self.idx_restrictions
            # record that we have considered this idx in state
            self.base_iter_idx += 1
            if i == iters_to_stop - 1:
                # before raising StopIteration we should void state
                self.base_iter = None
                self.base_iter_idx = None
                raise StopIteration
            else:
                iter_result = self.base_iter.__next__()
                if deliver_result:
                    return iter_result

    def __iter__(self):
        if self.base_iter_idx is not None:
            raise RuntimeError(
                f"Method: __iter__ was called on {self.__repr__()} before the previous iterator was completed."
            )
        # initialize
        self.base_iter = self.base_loader.__iter__()
        self.base_iter_idx = 0
        return self

    def __next__(self):
        if (self.base_iter is None) or (self.base_iter_idx == None):
            raise ValueError(
                "Cannot call next on LoaderRestrictor before calling iter on it."
            )

        iter_result = self.iterate_to_next_restricted_idx_or_raise_stop()
        feature = None
        label = None
        if self.subject_to_feature is not None:
            feature = self.subject_to_feature(subject_dict=iter_result)
        if self.subject_to_label is not None:
            label = self.subject_to_label(subject_dict=iter_result)

        if self.type_restrictions == "feature":
            iter_result = feature
        elif self.type_restrictions == "label":
            iter_result = label

        return iter_result

    def set_idx_restrictions(self, idx_restrictions):
        if not np.all(np.array(idx_restrictions) >= 0):
            raise ValueError(
                "Cannot set idx_restrictions to an array containing negative indices."
            )
        elif np.amax(idx_restrictions) > len(self.base_loader):
            raise ValueError(
                "Trying to set idx_restrictions with indices that exceed tha maximum range."
            )
        else:
            self.idx_restrictions = idx_restrictions

    def copy(self):
        return GaNDLFLoaderWrapper(parameters = self.parameters,
                                   train = self.train,
                                   csv_path = self.csv_path,
                                   type_restrictions = self.type_restrictions,
                                   idx_restrictions = self.idx_restrictions,
                                   subject_to_feature = self.subject_to_feature,
                                   subject_to_label = self.subject_to_label
                                   base_loader = self.base_loader)

    def __len__(self):
        return len(self.idx_restrictions)

    def __getitem__(self, indices):
        temp = self.copy()
        temp.idx_restrictions = temp.idx_restrictions[indices]
        return temp
    # TODO: Maybe we don't need this?
    def __reduce__(self):
        unpack = lambda info: GaNDLFLoaderWrapper(**info)
        packaged_info = {'parameters' : self.parameters,
                           'train' : self.train,
                           'csv_path' : self.csv_path,
                           'type_restrictions' : self.type_restrictions,
                           'idx_restrictions' : self.idx_restrictions,
                           'subject_to_feature' : self.subject_to_feature,
                           'subject_to_label' : self.subject_to_label,
                           'base_loader' : self.base_loader}
        return unpack, packaged_info


class GaNDLFPyTorchModel(PytorchModel):
    """
    Inherits from the PyTorchModel class, an interface to query a model without any assumption on how it is implemented.
    This particular class is to be used with pytorch models.
    """

    def __init__(self, model_obj, loss_fn, gandlf_config):
        """Constructor
        Args:
            model_obj: Model object.
            loss_fn: Loss function.
        """

        # Imports torch with global scope
        globals()['torch'] = __import__('torch')

        # Initializes the parent model
        super().__init__(model_obj, loss_fn)

        self.gandlf_config = gandlf_config

    def concatenated_logits(self, restricted_feature_loader):
        """Function to get the model output from restricted inputs.
        Args:
            restricted_feature_loader(GaNDLFLoaderWrapper): Wrapper for GaNDLF feature loader allowing slicing
        Returns:
            The concatenation of torch tensor model outputs over batches served up by the restricted feature loader.
        """
        with torch.no_grad():
            per_batch_logits = []
            for feature_batch in restricted_feature_loader:
                per_batch_logits.append(self.model_obj(feature_batch).to("cpu"))
            logits = torch.cat(per_batch_logits, dim=0)
        return logits

    def concatenated_labels(self, restricted_label_loader):
        """Function to get the concatenation of torch tensor batch labels.
        Args:
            restricted_loader (GaNDLFLoaderWrapper): Model input.
        Returns:
            The concatenation of torch tensor batch labels.
        """
        per_batch_labels = []
        for label_batch in restricted_label_loader:
            per_batch_labels.append(label_batch.to("cpu"))
        labels = torch.cat(per_batch_labels, dim=0)
        return labels

    def get_logits(self, restricted_feature_loader):
        """Alias of concatenated_outputs.
        """
    
        return self.concatenated_logits(restricted_feature_loader)

    def get_loss(self, restricted_feature_loader, restricted_label_loader, per_point=True):
        """Function to get the model loss on a given input and an expected output.
        Args:
            restricted_feature_loader (GaNDLFLoaderWrapper): Model input.
            restricted_label_loader (GaNDLFLoaderWrapper): Model expected output.
            per_point: Boolean indicating if loss should be returned per point or reduced.
        Returns:
            The loss value, as defined by the loss_fn attribute.
        """
        logits = self.concatenated_logits(restricted_feature_loader)
        labels = self.concatenated_labels(restricted_label_loader)
        if per_point:
            return self.loss_fn_no_reduction(logits,labels).detach().numpy()
        else:
            return self.loss_fn(logits, labels).item()

    def load_state_dict(self, state_dict):
        self.model_obj.load_state_dict(state_dict)

    def to(self, device):
        self.model_obj.to(device)

