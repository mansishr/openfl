import torch
from torch.utils.data import DataLoader
from torchio import DATA

from modified_images_from_data_frame import ImagesFromDataFrame
from GANDLF.utils.write_parse import get_dataframe
from GANDLF.utils import populate_header_in_parameters, parseTrainingCSV
from GANDLF.utils import one_hot, populate_channel_keys_in_params, get_class_imbalance_weights

def subject_to_feature(subject_dict, gandlf_config):
    features = torch.cat([subject_dict[key][DATA] for key in gandlf_config["channel_keys"]], 
                             dim=1).float().to(gandlf_config["device"])
    return features
    
def subject_to_label(subject_dict, gandlf_config):
    gt = subject_dict["label"]["data"].float().to(gandlf_config["device"])
    if gandlf_config["problem_type"] == "segmentation":
        gt = one_hot(gt, gandlf_config["model"]["class_list"])
    return gt




def get_train_loader(params, prevent_shuffling):
    """
    Get the training data loader.
    Args:
        params (dict): Dictionary of parameters.
    Returns:
        torch.utils.data.DataLoader: The training loader.
    """
    
    if prevent_shuffling:
        shuffle = False
    else:
        shuffle = True

    return DataLoader(
        ImagesFromDataFrame(
            get_dataframe(params["training_data"]),
            params,
            train=True,
            loader_type="train",
            prevent_shuffling=prevent_shuffling
        ),
        batch_size=params["batch_size"],
        shuffle=shuffle,
        pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
    )


def get_validation_loader(params, prevent_shuffling):
    """
    Get the validation data loader.
    Args:
        params (dict): Dictionary of parameters.
    Returns:
        torch.utils.data.DataLoader: The validation loader.
    """
    queue_from_dataframe = ImagesFromDataFrame(
        get_dataframe(params["validation_data"]),
        params,
        train=False,
        loader_type="validation",
        prevent_shuffling=prevent_shuffling
    )
    # Fetch the appropriate channel keys
    # Getting the channels for training and removing all the non numeric entries from the channels
    params = populate_channel_keys_in_params(queue_from_dataframe, params)

    return DataLoader(
        queue_from_dataframe,
        batch_size=1,
        shuffle=False,
        pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
    )


def get_single_loader(parameters, train, csv_path, prevent_shuffling):
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
        loader = get_train_loader(parameters, prevent_shuffling=prevent_shuffling)
        parameters["training_samples_size"] = len(loader)
        # Calculate the weights here
        (
            parameters["weights"],
            parameters["class_weights"],
        ) = get_class_imbalance_weights(parameters["training_data"], parameters)

    else:
        # get the validation loader
        loader = get_validation_loader(parameters, prevent_shuffling=prevent_shuffling)

    return (loader, parameters)
    
    
def get_loaders(parameters, prevent_shuffling, train_csv_path=None, val_csv_path=None):
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
                                                    csv_path=train_csv_path, 
                                                    prevent_shuffling=prevent_shuffling)
    val_loader, parameters = get_single_loader(parameters=parameters,
                                                train=False,
                                                csv_path=val_csv_path, 
                                                prevent_shuffling=prevent_shuffling)

    return (train_loader, val_loader, parameters)