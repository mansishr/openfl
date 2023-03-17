# This code is a modification of Challenge/Task_1/fets_challenge/gandlf_csv_adapter.py
# from the github project at https://github.com/FeTS-AI/Challenge
# the function: get_appropriate_file_paths_from_subject_dir was a cut and paist from
# Algorithms/fets/data/base_utils.py at https://github.com/FeTS-AI/Algorithms/blob/master/fets/data/base_utils.py

# Modifications performed by Brandon Edwards (Intel) and Mansi Sharma (Intel) and were
# primarily for the purpose of having test sets (was previously only train and val), in addition to creating sets from

# Note: The csvs creared in this modified script are per institution and there is a single csv for each of train, val, and test.
#       This differs from the single csv created by the original script (that was used by the FeTS Challenge code)

# Provided by the FeTS Initiative (www.fets.ai) as part of the FeTS Challenge 2022

# Combined authors (original and modifified script) in alphabetical order:
# Brandon Edwards (Intel)
# Patrick Foley (Intel)
# Mansi Sharma (Intel)
# Micah Sheller (Intel)

import os

import numpy as np
import pandas as pd

import argparse

def get_appropriate_file_paths_from_subject_dir(dir_path, 
                                                include_labels=False, 
                                                allowed_labelfile_endings=["_seg_binary.nii.gz", "_seg_binarized.nii.gz", "_SegBinarized.nii.gz", "_seg.nii.gz"], 
                                                excluded_labelfile_endings=[], 
                                                handle_missing_datafiles=False):
    '''
    This function takes a subject directory as input and return a dictionary of the full paths to the modalities (BraTS-specific)
    '''
    filesInDir = os.listdir(dir_path)

    # FIXME: There is more than one place the list below is defined (example: gandlf_data)
    # Move to one location and ensure sync with feature_modes from the flplan
    brats_modes = ['T1', 'T2', 'FLAIR', 'T1CE']
    label_tag = 'Label'
    # acceptable file endings for each scanning modality
    mode_to_endings = {'T1': ['_t1.nii.gz'], 
                      'T2': ['_t2.nii.gz'], 
                      'FLAIR': ['_flair.nii.gz'], 
                      'T1CE': ['_t1ce.nii.gz', '_t1gd.nii.gz']} 
    return_dict = {type: None for type in brats_modes}
    if include_labels:
        return_dict[label_tag] = None

    for _file in filesInDir:
        fpath = os.path.abspath(os.path.join(dir_path,_file))
        # Is this file a valid feature mode (and not already found)?
        for mode in brats_modes:
            if np.any([_file.endswith(ending) for ending in mode_to_endings[mode]]):
                if return_dict[mode] is None:
                    return_dict[mode] = fpath
                else:
                    raise RuntimeError('Found two {} files in {} '.format(mode, dir_path))
        if include_labels:
            # Is this file a valid label (and not alreay found or in the excluded labelfile list)
            allowed_label = np.any([_file.endswith(ending) for ending in allowed_labelfile_endings])
            excluded_label = np.any([_file.endswith(ending) for ending in excluded_labelfile_endings])
            if allowed_label and not excluded_label:
                if return_dict[label_tag] is None:
                    return_dict[label_tag] = fpath
                else:
                    raise RuntimeError('Found two label files (allowing any of {} and excluding any of {}) in directory {} '.format(allowed_labelfile_endings, excluded_labelfile_endings, dir_path))

    for key, value in return_dict.items():
        if value is None:
            if handle_missing_datafiles:
                print('\nNo {} file found in {}, but handle_missing_datafiles is True.\n'.format(key, dir_path))
                return None
            else:
                raise ValueError('No {} file found in {}, and handle_missing_datafiles is False.'.format(key, dir_path))
                 
    return return_dict


# some hard-coded keys
# feature stack order determines order of feature stack modes
# (so must be consistent across datasets used on a given model)
# dependency here with mode naming convention used in get_appropriate_file_paths_from_subject_dir
feature_modes = ['T1', 'T2', 'FLAIR', 'T1CE']
label_tag = 'Label'

mode_to_header_name = {'T1': 'Channel_0', 
                        'T2': 'Channel_1', 
                        'FLAIR': 'Channel_2', 
                        'T1CE': 'Channel_3'}

# hard coded samples used to train the FeTS2022 initial model
# and therefore best to keep out of split insitution validation
# set
init_train = ['FeTS2022_00159','FeTS2022_00172','FeTS2022_00187','FeTS2022_00199','FeTS2022_00211','FeTS2022_00221', \
              'FeTS2022_00235','FeTS2022_00243','FeTS2022_00258','FeTS2022_00269','FeTS2022_00282','FeTS2022_00291', \
              'FeTS2022_00300','FeTS2022_00311','FeTS2022_00321','FeTS2022_00332','FeTS2022_00344','FeTS2022_00353', \
              'FeTS2022_00370','FeTS2022_00380','FeTS2022_00391','FeTS2022_00403','FeTS2022_00413','FeTS2022_00425', \
              'FeTS2022_00440','FeTS2022_01000','FeTS2022_01038','FeTS2022_01046','FeTS2022_01054','FeTS2022_01062', \
              'FeTS2022_01070','FeTS2022_01078','FeTS2022_01086','FeTS2022_01094','FeTS2022_01102','FeTS2022_01110', \
              'FeTS2022_01118','FeTS2022_01126','FeTS2022_01134','FeTS2022_01205','FeTS2022_01213','FeTS2022_01221', \
              'FeTS2022_01229','FeTS2022_01237','FeTS2022_01245','FeTS2022_01253','FeTS2022_01261','FeTS2022_01269', \
              'FeTS2022_01277','FeTS2022_01293','FeTS2022_01307','FeTS2022_01315','FeTS2022_01323','FeTS2022_01331', \
              'FeTS2022_01339','FeTS2022_01347','FeTS2022_01355','FeTS2022_01363','FeTS2022_01371','FeTS2022_01379', \
              'FeTS2022_01387','FeTS2022_01395','FeTS2022_01403']


def train_val_test_split(subdirs, 
                         train_val_test, 
                         shuffle=True):
    """
    train_val_test is a list of three floats (summing to one) that determine the portions
    going to train, val, and test respectively
    Assumption: None of the entries in train_val_test are 0.0
    """  

    def validate_portions(split, 
                          remains, 
                          total, 
                          leave, 
                          portion):
        """
        Sanity checks used to ensure take_portion_leave_some function does what is intended
        """

        # if we we left only the minimal number of samples, was this necessary
        if len(remains) == leave:
            if (int(portion * len(total)) + leave + 1 <= len(total)):
                raise ValueError(f"Something is wrong, {len(split)} were split off during take_portion_... which were not enough.") 
        # if we were able to leave more than minimum, split should be close to correct size
        elif abs(portion * len(total) - len(split)) > 1:
            raise ValueError(f"Split during take_portion_... targeted pulling off {portion * len(total)} samples and took {len(split)} instead, off by more than 1 which is unexpected since more could have been taken given the value of leave.")
        
        # Split must have at least one element, and remains must have at least -leave- elements
        if len(split) == 0:
            raise ValueError(f"List split off during take_portion_... must have at least one element.")
        if len(remains) < leave:
            raise ValueError(f"Remains of split during take_portion_... must have at least leave={leave} number of elements and it has {len(remains)} instead.")
        
        # The assumption here is that total is a list with no repeat entries (used in two checks after this one)
        if len(total) != len(set(total)):
            raise ValueError(f"List being split has repeat entries which could be tolerated but was not coded for in sanity checks")
        # given no repeat entries, the following checks all entries were accounted for after split.
        if set(split).union(set(remains)) != set(total):
            raise ValueError(f"Something is wrong in list split logic, entries are missing or added after split.")
        if set(split).intersection(set(remains)) != set():
            raise ValueError(f"Something is wrong in list split logic, split and remains share entries.")
        return

    def take_portion_leave_some(total_list, 
                                portion, 
                                leave, 
                                shuffle=True):
        """
        Take a -portion- (float strickly between 0 and 1) of -total_list- and return that and the
        ramains of the list. Make sure that at least one entry from -total_list- is taken and
        make sure that -leave- number of samples sit in the remains. If any of this is not possible
        throw an exception.
        """

        if portion <= 0.0 or portion >= 1.0:
            raise ValueError(f"portion must be strictly between 0.0 and 1.0")
        if leave < 0:
            raise ValueError(f"leave must be non-negative")

        if shuffle:
            np.random.shuffle(total_list)
        n_entries = len(total_list)
        if n_entries  - 1 < leave:
            raise ValueError(f"A list of {n_entries} entries was provided, to split off at least one but leave {leave}. This is not possible.")
        
        cutpoint = int(n_entries * portion)
        # adjust to acomodate requirements
        if cutpoint == 0:
            cutpoint = 1
        if n_entries - cutpoint < leave:
            cutpoint = n_entries - leave
            if cutpoint < 1:
                raise ValueError(f"Unexpected result, go in and redo the math where this code lies.")

        split = total_list[:cutpoint]
        remains = total_list[cutpoint:]

        validate_portions(split=split, 
                        remains=remains, 
                        total=total_list, 
                        leave=leave, 
                        portion=portion)

        return split, remains

    percent_train = train_val_test[0]
    percent_val = train_val_test[1]
    percent_test = train_val_test[2]

    if len(train_val_test) != 3:
        raise ValueError(f"argument train_val_test should be a list of three floats")
    if np.sum(train_val_test) != 1.0:
        raise ValueError(f"The sum of the three floats in train_val_test should be 1.0")
    if (percent_train <= 0) or (percent_val <= 0) or (percent_test <= 0):
        raise ValueError(f"All of the entries in train_val_test must be positive")


    # we don't want to train, val, or test on samples that were used to train the initial model
    unwanted_subdirs = [subdir for subdir in subdirs if subdir in init_train]
    # limit subdirs to those that do not lie in init_train
    wanted_subdirs = [subdir for subdir in subdirs if subdir not in init_train]

    assert len(unwanted_subdirs) + len(wanted_subdirs) == len(subdirs)

    n_trainvaltest = len(wanted_subdirs)

    if n_trainvaltest < 3:
        raise ValueError(f"At least three samples must remain after removing samples used to train the initial model.")
    
    train_subdirs, valtest_subdirs = take_portion_leave_some(total_list=wanted_subdirs, 
                                                             portion=percent_train, 
                                                             leave=2)
    
    val_subdirs, test_subdirs = take_portion_leave_some(total_list=valtest_subdirs, 
                                                        portion=percent_val/(percent_val+percent_test), 
                                                        leave=1)

    if shuffle:
        np.random.shuffle(train_subdirs)
        np.random.shuffle(val_subdirs)
        np.random.shuffle(test_subdirs)
    
    return train_subdirs, val_subdirs, test_subdirs


def paths_dict_to_dataframes(paths_dict):
    
    dataframes = {}
    
    for inst_name, inst_paths_dict in paths_dict.items():
        dataframes[inst_name] = {}
        for usage in ['train', 'val', 'test']:
            channel_headers = [f'Channel_{idx}' for idx in range(len(feature_modes))]
            # intitialize columns
            columns = {'SubjectID': [], 
                       label_tag: [] 
                       }
            columns.update({mode_to_header_name[mode]: [] for mode in feature_modes})

            for key_to_fpath in inst_paths_dict[usage]:
                columns['SubjectID'].append(key_to_fpath['SubjectID']) 
                columns[label_tag].append(key_to_fpath[label_tag]) 
                for mode in feature_modes:
                    columns[mode_to_header_name[mode]].append(key_to_fpath[mode])

            dataframes[inst_name][usage] = pd.DataFrame(columns, dtype=str)
    return dataframes
    

def construct_fedsim_csvs(source_data_pardir, 
                         split_subdirs_path, 
                         train_val_test, 
                         csv_pardir):
    """
    pardir (str): full path to the parent directory of all samples
    split_subdirs_path (str): full path to the institutional split csv
    train_val_test (list of 3 floats suming to 1.0): The portions going to train, val, test respectively after removing samples used to train intitial model
    federated_simulation_train_val_test_csv_path (str): full path to the directory that will hold the train, val, and test csvs per institution
    """
    
    # read in the csv defining the subdirs per institution
    split_subdirs = pd.read_csv(split_subdirs_path, dtype=str)
    
    if not set(['Partition_ID', 'Subject_ID']).issubset(set(split_subdirs.columns)):
        raise ValueError("The provided csv at {} must at minimum contain the columns 'Partition_ID' and 'Subject_ID', but the columns are: {}".format(split_subdirs_path, list(split_subdirs.columns)))
    
    # sanity check that all subdirs provided in the dataframe are unique
    if not split_subdirs['Subject_ID'].is_unique:
        raise ValueError("Repeated references to the same data subdir were found in the 'Subject_ID' column of {}".format(split_subdirs_path))
    
    train_val_specified = ('TrainOrVal' in split_subdirs.columns)
    if train_val_specified:
        raise ValueError(f"GaNDLF csv construction script not intended for use on csvs that already specify train or val per sample. Can change this easily, but changes are needed.")
    
    inst_names = list(split_subdirs['Partition_ID'].unique())
    
    paths_dict = {inst_name: {'train': [], 'val': [], 'test': []} for inst_name in inst_names}
    for inst_name in inst_names:
        
        subdirs = list(split_subdirs[split_subdirs['Partition_ID']==inst_name]['Subject_ID'])
        train_subdirs, val_subdirs, test_subdirs = train_val_test_split(subdirs=subdirs, 
                                                                        train_val_test=train_val_test)
        # NOTE: There is a change in convention going on here, instituional split csv uses
        #       key 'Subject_ID' and we use here 'SubjectID' because that is what GaNDLF will want

        for subdir in train_subdirs:
            inner_dict = get_appropriate_file_paths_from_subject_dir(os.path.join(source_data_pardir, subdir), include_labels=True)
            inner_dict['SubjectID'] = subdir
            paths_dict[inst_name]['train'].append(inner_dict)
            
        for subdir in val_subdirs:
            inner_dict = get_appropriate_file_paths_from_subject_dir(os.path.join(source_data_pardir, subdir), include_labels=True)
            inner_dict['SubjectID'] = subdir
            paths_dict[inst_name]['val'].append(inner_dict)

        for subdir in test_subdirs:
            inner_dict = get_appropriate_file_paths_from_subject_dir(os.path.join(source_data_pardir, subdir), include_labels=True)
            inner_dict['SubjectID'] = subdir
            paths_dict[inst_name]['test'].append(inner_dict)
        
    # now construct the dataframes
    dataframes =  paths_dict_to_dataframes(paths_dict=paths_dict)

    # now save dataframes to csv (three - train, val, test - per institution)
    for inst_name, dfs in dataframes.items():
        train_df = dfs['train']
        val_df = dfs['val']
        test_df = dfs['test']

        inst_csv_dir = os.path.join(csv_pardir, inst_name)
        # This will be a little annoying, but I will raise and exception if these directories already exist
        os.makedirs(inst_csv_dir, exist_ok=False)

        train_df_path = os.path.join(inst_csv_dir, inst_name + "_train.csv")
        val_df_path = os.path.join(inst_csv_dir, inst_name + "_val.csv")
        test_df_path = os.path.join(inst_csv_dir, inst_name + "_test.csv")

        train_df.to_csv(train_df_path, index=False)
        val_df.to_csv(val_df_path, index=False)
        test_df.to_csv(test_df_path, index=False)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--source_data_pardir',
        type=str,
        help='Absolute path to the parant directory holding the per subject directories')
    argparser.add_argument(
        '--split_subdirs_path',
        type=str,
        help='Absolute path to the institutional split csv')
    argparser.add_argument(
        '--train_part',
        type=float,
        help='weight of train part of data split')
    argparser.add_argument(
        '--val_part',
        type=float,
        help='weight of val part of data split')
    argparser.add_argument(
        '--test_part',
        type=float,
        help='weight of test part of data split')
    argparser.add_argument(
        '--csv_pardir',
        type=str,
        help="Absolute path to the output directory that will hold the final csvs.")
    
    args = argparser.parse_args()

    kwargs = vars(args)

    train_val_test = [kwargs.pop('train_part'), kwargs.pop('val_part'), kwargs.pop('test_part')]

    kwargs['train_val_test'] = train_val_test

    construct_fedsim_csvs(**kwargs)
