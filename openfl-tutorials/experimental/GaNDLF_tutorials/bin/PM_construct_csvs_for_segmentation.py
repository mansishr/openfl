import pandas as pd 
import os, argparse
import numpy as np
from pathlib import Path

"""
This module.main takes the original train and test csvs and splits them in order to create PM_(other(pop and ref), train, and test) 
csvs to be used for privacy meter evaluation. The PM_other=PM_pop set is used for the population set in the case of the 
population attack and PM_other=PM_ref is used for the reference set (used for training reference models) in the case of 
the reference attack. This version ignores class information since it is being used for segmentation data.

Here are some requirements
PM_train and PM_test should be of the same size(we test the membership inference attack using these)
PM_train should consist onl of samples used to train the target model
PM_test should not hold any samples (or relate to subject IDs from other data) seen by the model (including validation sets)
PM_pop should not have samples or the same subject IDs as samples used to train the target model
PM_ref should not have any samples sharing subject ID ith PM_train and PM_test, but can hold samples used to 
train or validate the target model otherwise

Basic idea for the script: 
Take orig_test and split it to create PM_test and PM_pop (separate subject IDs)

Split off samples from orig_train to create a PM_train the same size as PM_test
Samples from orig_train with different subject ID from the above split off go with orig_val to
combine with PM_pop and form a PM_ref with similar class balance as others

Handling of subject IDs here is really not an issue since samples have distinct subject IDs for the BraTS
data we are using. We keep the subject ID check however in ordre to minimally change the original script.

Assumption: All rows of dataframes here are associated with distinct subject IDs

More notes:
The original test set is split into PM_pop and PM_test set. 
The orig_train is sampled to get PM_train. 
Any samples left in orig_train (so not sharing subject ID with samples pulled to go into PM_train) are 
put into a new dataset with the PM_pop and orig_val (and rebalanced if needed) to create PM_ref. 
privacy meter train and test sets are of equal sizes.  

"""

## some hard-coded choices
allow_dropped_orig_test_samples = False
allow_dropped_orig_train_samples = True


def drop_one_if_needed(df_1, df_2):
    """
    Checks to be sure the df lengths are only off by one, and if so makes them equal in size by dropping a row
    """
    l1 = len(df_1)
    l2 = len(df_2)
    if abs(l1 - l2) > 1:
        raise ValueError(f"drop_one_if_needed called on dataframes of sizes {l1} and {l2} (should have been off by at most one)")
    elif l1 > l2:
        print(f"Dropping one row to make two dfs equal length.")
        return df_1[1:], df_2
    elif l2 > l1:
        print(f"Dropping one row to make two dfs equal length.")
        return df_1, df_2[1:]
    else:
        # here l1=l2
        return df_1, df_2
    
def check_ids(df):
    # check assumption that subjectid is distinct for all rows of df
    if len(df) != len(df["subjectid"].unique()):
        raise ValueError(f"Assumption of distinct subject ID for each row is broken - df has {len(df)} rows and there are only {df['subjectid'].unique()} subject IDs represented.")


def basic_split(df, left_split_frac):
    num_samples = len(df)
    cutpoint = int(left_split_frac * num_samples)
    left_split = df[:cutpoint].sample(frac=1.0)
    right_split = df[cutpoint:].sample(frac=1.0)

    # some checks
    if set(left_split.index).union(set(right_split.index)) != set(df.index):
        raise RuntimeError("basic_split did not preserve the index set!")
    if len(left_split) + len(right_split) != len(df):
        raise RuntimeError("basic_split did not preserve total row count!")
    return left_split, right_split


def construct_PM_csvs(orig_train_csv_path, 
                      orig_val_csv_path, 
                      orig_test_csv_path, 
                      new_csv_folder, 
                      data_name, 
                      orig_test_portion_to_pop): 

    Path(new_csv_folder).mkdir(parents=True, exist_ok=True)
    
    orig_train = pd.read_csv(orig_train_csv_path)
    orig_val = pd.read_csv(orig_val_csv_path)
    orig_test = pd.read_csv(orig_test_csv_path)

    # column names are now case-insensitive
    orig_train.columns = orig_train.columns.str.lower()
    orig_val.columns = orig_val.columns.str.lower()
    orig_test.columns = orig_test.columns.str.lower()

    # Check that subject IDs are distinct across all three dataframes
    check_ids(pd.concat([orig_train, orig_val, orig_test], axis=0))

    # We shuffle here and many other places in this script
    orig_train = orig_train.sample(frac=1).reset_index(drop=True)
    orig_val = orig_val.sample(frac=1).reset_index(drop=True)
    orig_test = orig_test.sample(frac=1).reset_index(drop=True)

    # print out some info
    print(f"\nOriginal train had a count of {len(orig_train)}")
    print(f"Original val had a count of {len(orig_val)}")
    print(f"Original test had a count of {len(orig_test)}\n")
    
    # split test to create PM_test and PM_pop (PM_ref will be created using PM_pop)
    PM_pop, PM_test = basic_split(orig_test, orig_test_portion_to_pop)

    print(f"\nPM_pop is now set with a size of: {len(PM_pop)}")
    print(f"PM_test is currently size: {len(PM_test)} but may need to change to account for training set size.\n")

    
    # PM_test and PM_train need to be of equal size
    if len(PM_test) > len(orig_train):
        if not allow_dropped_orig_test_samples:
            raise ValueError(f"Not allowing to drop test samples but we do not have enough training samples to match fraction of samples going to PM_test from orig_test.")
        else:
            PM_test, _ = basic_split(PM_test,left_split_frac=len(orig_train)/float(len(PM_test)))
            PM_train = orig_train
    else:
        if not allow_dropped_orig_train_samples:
            raise ValueError(f"Not allowing to drop train samples but we do not have enough testing samples after pulling off samples for PM_other.")
        else:
            PM_train, samples_split_from_orig_train = basic_split(orig_train, left_split_frac=len(PM_test)/float(len(orig_train)))   

    # PM_pop, samples split from orig_train, and orig_val combine to create PM_ref (maybe slightly off in balance after combining) 
    
    # This is a long accounting to ensure we carefully add samples to form PM_ref in priority order
    # (PM_pop are best, followed by orig_val then samples_split_from_orig_train)
    if len(orig_train) < len(PM_pop):
        PM_ref, _ = basic_split(df=PM_pop, 
                                left_split_frac=len(orig_train)/float(len(PM_pop)))
    elif len(orig_train) == len(PM_pop):
        PM_ref = PM_pop
    elif len(orig_train) < len(PM_pop) + len(orig_val):
        num_needed_from_orig_val = len(orig_train) - len(PM_pop)
        samples_split_from_orig_val, _ = basic_split(df=orig_val, 
                                                     left_split_frac=num_needed_from_orig_val/float(len(orig_val)))
        PM_ref = pd.concat([PM_pop, samples_split_from_orig_val])
    else: 
        PM_ref = pd.concat([PM_pop, orig_val])
    
    # check that PM_test and PM_train are the same length, if only off by one then handle it
    PM_test, PM_train = drop_one_if_needed(PM_test, PM_train)
   
    ###############################
    # Now some final checks
    ###############################

    orig_train_length = len(orig_train)
    orig_val_length = len(orig_val)
    orig_test_length = len(orig_test)

    final_pm_train_length = len(PM_train)
    final_pm_test_length = len(PM_test)
    final_pm_pop_length = len(PM_pop)
    final_pm_ref_length = len(PM_ref)

    print(f"\nTrain, Test, Pop, Ref -- Counts: {final_pm_train_length, final_pm_test_length, final_pm_pop_length, final_pm_ref_length}")
    
    # PM_test and PM_train need to be the same length
    if final_pm_train_length != final_pm_test_length:
        raise ValueError("Final PM_train len: {final_pm_train_length} is not equal to final PM_test len: {final_pm_test_length}")
    
    PM_train_csv_path = os.path.join(new_csv_folder, data_name + "_pm_train.csv") 
    PM_test_csv_path = os.path.join(new_csv_folder, data_name + "_pm_test.csv") 
    PM_pop_csv_path = os.path.join(new_csv_folder, data_name + "_pm_pop.csv")
    PM_ref_csv_path = os.path.join(new_csv_folder, data_name + "_pm_ref.csv")

    # now shuffle the resulting dataframes once more
    PM_train = PM_train.sample(frac=1).reset_index(drop=True)
    PM_test = PM_test.sample(frac=1).reset_index(drop=True)
    PM_pop = PM_pop.sample(frac=1).reset_index(drop=True)
    PM_ref = PM_ref.sample(frac=1).reset_index(drop=True)

    PM_train.to_csv(PM_train_csv_path, index=False)
    PM_test.to_csv(PM_test_csv_path, index=False)
    PM_pop.to_csv(PM_pop_csv_path, index=False)
    PM_ref.to_csv(PM_ref_csv_path, index=False)
