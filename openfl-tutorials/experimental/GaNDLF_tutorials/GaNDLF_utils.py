import numpy as np

import torch
from torchio import DATA


from GANDLF.models import global_models_dict

from privacy_meter.model import PytorchModel

from get_loaders import get_train_loader, get_validation_loader, get_single_loader
   

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

def consistent_loader(loader, num_attempts, num_subjects, verbose=True):
    chnl1_tensors = []
    subject_ids = []

    all_equal = True

    for a_idx, attempt in enumerate(range(num_attempts+1)):
        if a_idx == 0:
            for s_idx, subject in enumerate(loader):
                if s_idx == num_subjects:
                    break
                else:
                    chnl1_tensors.append(subject['1']['data'])
                    subject_ids.append(subject['subject_id'])
        else:
            print(f"Comparing one run of base loader with another...attempt={a_idx+1}")
            equal = True
            for s_idx, subject in enumerate(loader):
                if s_idx == num_subjects:
                    break
                else:
                    if not torch.equal(chnl1_tensors[s_idx], subject['1']['data']) or subject_ids[s_idx] != subject['subject_id']:
                        if verbose:
                            tensor_diff_idxs = ~(chnl1_tensors[s_idx] == subject['1']['data'])
                            first_time = chnl1_tensors[s_idx][tensor_diff_idxs]
                            second_time = subject['1']['data'][tensor_diff_idxs]
                            print(f"\nGot a difference in what loader produced:")
                            print("--- Subjects ---")
                            print(f"FIRST TIME: {subject_ids[s_idx]}")
                            print(f"SECOND TIME: {subject['subject_id']}\n")
                            print("--- Part Tensors ---")
                            print(f"FIRST TIME: {first_time}")
                            print(f"SECOND TIME: {second_time}\n\n")
                        equal = False
            if not equal: 
                all_equal = False
    return all_equal


# Help GaNDLF loaders be treated like numpy arrays (slicing). Also, help deepcopy GaNDLF loaders (via __reduce__)
class GaNDLFLoaderWrapper(object):

    # Some hard coded choices as to how many times to run 
    # and how many subjects to check against when testing that the base loader
    # for that it produces the same data over multiple usages
    num_attempts = 5
    num_subjects = 5

    def __init__(self, 
                 parameters, 
                 train, 
                 type_restrictions, 
                 subject_to_feature, 
                 subject_to_label,
                 idx_restrictions=None,
                 csv_path = None, 
                 base_loader=None, 
                 num_attempts=num_attempts,
                 num_subjects=num_subjects):
        """
        TODO rewrite this documentation below-----
        restriction (tuple of: str, np.ndarray): First component can be 'feature', 'label', or
        'feature_and_label', array specifies which indices to allow during iteration. Note base loader
        must load exactly the same upon each usage. We insert a test for this reproducibility below.
        """
        super().__init__()
            
        self.parameters = parameters
        self.train = train
        self.type_restrictions = type_restrictions
        self.idx_restrictions = idx_restrictions
        self.subject_to_feature = subject_to_feature
        self.subject_to_label = subject_to_label
        self.type_restrictions = type_restrictions 
        self.idx_restrictions = idx_restrictions
        self.csv_path = csv_path
        self.base_loader = base_loader
        
        if self.base_loader is None:
            self.base_loader, self.parameters = get_single_loader(parameters=self.parameters, 
                                                 train=self.train, 
                                                 csv_path=self.csv_path, 
                                                 prevent_shuffling=True)
            
        # Try to catch the base loader breaking the assumption of reproducibility
        # (this check is specific to BraTS) - only checks subject_id and 1st channel, 
        # so it is possible that the fact of it not loading consistently is not catched
        
        if not consistent_loader(loader=self.base_loader,
                                 num_attempts=num_attempts, 
                                 num_subjects=num_subjects):
            raise ValueError(f"Base GaNDLF loader is not deterministic and so loader wrapper will not work!")
        else:
            print(f"Base loader sent to GaNDLFLoaderWrapper appeared to be deterministic when tested against {num_attempts} attempts checking only channel 1 of {num_subjects} subjects.")        

        self.base_loader_length = len(self.base_loader)       
        # some parameter handling
        if self.idx_restrictions is None:
            self.idx_restrictions = np.arange(len(self.base_loader))

        # sanity check arguments
        if self.type_restrictions not in ["feature", "label", "feature_and_label"]:
            raise ValueError(
                "The first element of the restrictions tuple must be 'feature', 'label', or 'feature_and_label'."
            )
        if not base_loader and not csv_path:
            raise ValueError(f"Exactly one of base_loader and csv_path should not be None.")
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
                                   type_restrictions = self.type_restrictions,
                                   idx_restrictions = self.idx_restrictions,
                                   subject_to_feature = self.subject_to_feature,
                                   subject_to_label = self.subject_to_label,
                                   csv_path = self.csv_path,
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
                           'type_restrictions' : self.type_restrictions,
                           'idx_restrictions' : self.idx_restrictions,
                           'subject_to_feature' : self.subject_to_feature,
                           'subject_to_label' : self.subject_to_label,
                           'csv_path' : self.csv_path,
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

        self.model_obj = model_obj
        self.loss_fn = loss_fn
        self.gandlf_config = gandlf_config

    def get_logits(self, restricted_feature_loader):
        """Alias of concatenated_outputs.
        """
        raise NotImplementedError(f"get_logits not implemented for this segmentation model")

    def get_loss(self, restricted_feature_loader, restricted_label_loader, per_point=True):
        """Function to get the model loss on a given input and an expected output.
        Args:
            restricted_feature_loader (GaNDLFLoaderWrapper): Model input.
            restricted_label_loader (GaNDLFLoaderWrapper): Model expected output.
            per_point: Boolean indicating if loss should be returned per point or reduced.
        Returns:
            The loss value, as defined by the loss_fn attribute.

        NOTE: Here we rely on the data loaders to have batch size of 1.
        """

        # validate that loaders being used have batch size of 1
        for idx, (features, labels) in enumerate(zip(restricted_feature_loader, restricted_label_loader)):
            if idx == 0:
                if features.shape[0] != 1:
                    raise ValueError(f"feature batch is not 1 (requirement not met in wrapped_model.get_loss)!")
            elif labels.shape[0] != 1:
                raise ValueError(f"label batch is not 1 (requirement not met in wrapped_model.get_loss)!")
            break

        losses = []

        for features, labels in zip(restricted_feature_loader, restricted_label_loader):
            prediction = self.model_obj(features)
            losses.append(self.loss_fn(pm=prediction, gt=labels))

        if per_point:
            return torch.cat(losses, dim=0).detach().numpy()
        else:
            return torch.mean(torch.Tensor(losses), dim=0)

    def load_state_dict(self, state_dict):
        self.model_obj.load_state_dict(state_dict)

    def to(self, device):
        self.model_obj.to(device)




