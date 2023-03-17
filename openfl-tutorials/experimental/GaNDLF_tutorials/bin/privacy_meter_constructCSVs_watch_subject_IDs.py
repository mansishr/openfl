import pandas as pd 
import os, argparse
import numpy as np
from pathlib import Path

"""
This module.main takes the original train and test csvs and splits them in order to create PM_(other(pop and ref), train, and test) 
csvs to be used for privacy meter evaluation. The PM_other=PM_pop set is used for the population set in the case of the 
population attack and PM_other=PM_ref is used for the reference set (used for training reference models) in the case of 
the reference attack.

Here are some requirements
PM_train and PM_test should be of the same size(we test the membership inference attack using these)
PM_train should consist onl of samples used to train the target model
PM_test should not hold any samples (or relate to subject IDs from other data) seen by the model (including validation sets)
PM_pop should not have samples or the same subject IDs as samples used to train the target model
PM_ref should not have any samples sharing subject ID ith PM_train and PM_test, but can hold samples used to 
train or validate the target model otherwise
All sets should ideally have the same class imbalance.

Basic idea for the script: 
Take orig_test and split it to create PM_test and PM_pop (separate subject IDs) and similar class balance
as the originals
Split off samples from orig_train to create a PM_train the same size as PM_test and similar class balance
Samples from orig_train with different subject ID from the above split off go with orig_val to
combine with PM_pop and form a PM_ref with similar class balance as others


Assumptions:
- orig_train, orig_val, and orig_test have very similar class balance
- WHEN SPLITTING ANY OF THESE THREE ON SUBJECT ID, THE CLASS BALANCE IS SIMILAR !!!!!

More notes:
We want all of PM train, test, and pop datasets to have the same class balance (we are dealing with 
unbalanced data here) and to have the
same imalance as the original test data for apples to apples comparison of model accuracy and leakage
(leakage being measured using PM_train and PM_test). We want PM_pop to again have the same class
imbalance so that the thresholds computed using the loss value distributions we get over it are appropriate
for the PM_train and PM_test sets we will use them for. We will want PM_ref to have the same class imbalance as 
orig_train so that the trained reference models match as closely as possible to the training of the target model.
 PM_ref will consist of PM_pop with some orig train and orig val. All the time trying to split by subject ID so that
 the same subject ID does not land in two PM sets (except PM_pop and PM_ref can share samples and subject IDs)

The original test set is split into PM_pop and PM_test sets (again maintaining class balance and splitting
on subject ID). 
The orig_train is sampled to get PM_train with the same class imbalance and same size as PM_test. 
Any samples left in orig_train (so not sharing subject ID with samples pulled to go into PM_train) are 
put into a new dataset with the PM_pop and orig_val (and rebalanced if needed) to create PM_ref. 
privacy meter train and test sets are of equal sizes and all have the same class balance.  

NOTE: Assuming here binary classification for simplicity, a check will throw an exception if this is not the case.

"""

## some hard-coded choices
allow_dropped_orig_test_samples = False
allow_dropped_orig_train_samples = True


def split_by_class(df):
    class_zero_df = df[df['valuetopredict']==0]
    class_one_df = df[df['valuetopredict']==1]
    return {0: class_zero_df, 1: class_one_df}

def split_by_subject_id(df):
    subject_ids = df['subject_id'].unique()
    by_id_dict = {}
    total_count = 0
    for id in subject_ids:
        this_df = df[df['subject_id']==id]
        by_id_dict[id] = this_df
        total_count += len(this_df)
    assert total_count == len(df)
    return by_id_dict
    

def parse_classes(df):
    perclass_info = split_by_class(df)
    perclass_info.update({'Counts': {0:len(perclass_info[0]), 
                                    1: len(perclass_info[1])},
                         'Balance': float(len(perclass_info[0])) / (len(perclass_info[0]) + len(perclass_info[1]))
                         })  
    return perclass_info


def get_ids(df):
    return df['subject_id'].unique()


def ids_distinct(df_list):
    distinct = True
    for idx, df in enumerate(df_list):
        for inner_idx in range(len(df_list)):
            if inner_idx != idx:
                if set(get_ids(df)).intersection(set(get_ids(df_list(inner_idx)))) != set():
                    distinct = False
    return distinct


def rebalance(df, target_balance):
    if target_balance < 0.0 or target_balance > 1.0:
        raise ValueError("Target balance should never be below 0 or above 1.")

    perclass_info = parse_classes(df)
    if perclass_info['Balance'] < target_balance:
        drop_1_cut = int(perclass_info['Counts'][0] + perclass_info['Counts'][1] -  perclass_info['Counts'][0]/target_balance)
        new_df = pd.concat([perclass_info[0], perclass_info[1][drop_1_cut:]]).sample(frac=1.0).reset_index(drop=True)
        dropped_1 = perclass_info[1][:drop_1_cut]
        dropped_dict = {0: [], 1: dropped_1}
    else:
        drop_0_cut = int(perclass_info['Counts'][0] + perclass_info['Counts'][1] * (target_balance/(target_balance-1)))
        new_df = pd.concat([perclass_info[0][drop_0_cut:], perclass_info[1]]).sample(frac=1.0).reset_index(drop=True)
        dropped_0 = perclass_info[0][:drop_0_cut]
        dropped_dict = {0: dropped_0, 1: []}
    # test
    new_balance = parse_classes(new_df)['Balance']
    error_bound = 2/len(df)
    if abs(new_balance - target_balance)>error_bound:
        raise ValueError("re_balance did not get target balance with new_balance: {new_balance}, target_balance: {target_balance} and error bound: {error_bound}")
    return new_df, dropped_dict


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


def basic_stratified_split(df, left_split_frac):
    perclass_info = parse_classes(df)
    class_0_cutpoint = int(left_split_frac * perclass_info['Counts'][0])
    class_1_cutpoint = int(left_split_frac * perclass_info['Counts'][1])
    left_split = pd.concat([perclass_info[0][:class_0_cutpoint], perclass_info[1][:class_1_cutpoint]]).sample(frac=1.0)
    right_split = pd.concat([perclass_info[0][class_0_cutpoint:], perclass_info[1][class_1_cutpoint:]]).sample(frac=1.0)

    # some checks
    if set(left_split.index).union(set(right_split.index)) != set(df.index):
        raise RuntimeError("basic_stratified_split did not preserve the index set!")
    if len(left_split) + len(right_split) != len(df):
        raise RuntimeError("basic_stratified_split did not preserve total row count!")
    left_balance = parse_classes(left_split)['Balance']
    right_balance = parse_classes(right_split)['Balance']
    df_balance = parse_classes(df)['Balance']
    if abs(left_balance - df_balance) > 1/len(left_split):
        raise ValueError(f"basic_stratified_split did not keep close balance with left_balance being: {left_balance}, df_balance being: {df_balance} and the error tolerance being 1/len(left): {1/len(left_split)}")
    if abs(right_balance - df_balance) > 1/len(right_split):
        raise ValueError(f"basic_stratified_split did not keep close balance with right_balance being: {right_balance}, df_balance being: {df_balance} and the error tolerance being 1/len(right): {1/len(right_split)}")

    return left_split, right_split


def approx_optimal_by_subject_split(df, left_split_frac, attempts=2000):
    """
    Split by subject id trying many random splits (but not all) and selecting one giving closest to split frac
    CRITICAL ASSUMPTION: we are relying on similar class imbalance for all subject id shards 
    (we do not track these when determining the best split among the attempt number of random ones explored)
    """
    total_samples = len(df)
    subject_id_split = split_by_subject_id(df)
    subject_ids = list(subject_id_split.keys())
    # we will track size of each subject id shard, 
    sizes = {}
    for id in subject_ids:
        sizes[id] = len(df[df['subject_id']==id])

    # right error is the same as left error (initial error is left_split_frac-0)    
    best_error = left_split_frac
    best_left_split_ids = []

    # each attempt shuffles the subject ids and finds the cut point of that array giving closest to split frac
    for attempt in range(attempts):
        np.random.shuffle(subject_ids)
        left_split_size = 0
        left_split_ids = []
        # right error is the same as left error (initial error is left_split_frac-0)
        best_error_this_attempt = left_split_frac
        for idx, id in enumerate(subject_ids):
            left_split_size += sizes[id]
            diff = left_split_frac - left_split_size/total_samples

            if diff >= 0:
                # sanity check
                if diff > best_error_this_attempt:
                    raise ValueError("Unexpected non-decrease in error by taking a step down!!!")
                else:
                    best_error_this_attempt = diff
                    left_split_ids.append(id)
            else:
                # here not clear whether the potential absolute error is better or not
                if np.absolute(diff) > best_error_this_attempt:
                    # got worse and will only get more worse for this shuffling
                    break
                else:
                    # got better but will get worse from here for this shuffling
                    best_error_this_attempt = np.absolute(diff)
                    left_split_ids.append(id)
                    break
        if best_error_this_attempt < best_error:
            best_error = best_error_this_attempt
            best_left_split_ids = left_split_ids
    
    best_right_split_ids = list(set(subject_ids).difference(set(best_left_split_ids)))

    # sanity checks (we got all ids & non-overlapping)
    assert set(best_left_split_ids).union(set(best_right_split_ids)) == set(subject_ids)
    assert set(best_left_split_ids).intersection(set(best_right_split_ids)) == set()

    # now put together the preliminary split with the best option you found 
    left_split = df[df['subject_id'].isin(best_left_split_ids)]
    right_split = df[df['subject_id'].isin(best_right_split_ids)]

    print("\n########################################################################################")
    print(f"Completing approximate optimal by subject split with target left_split_fraction of {left_split_frac}")
    print(f"Error in splitting by subject ID was reduced to: {best_error}")
    print("########################################################################################\n")

    return left_split, right_split
                    

def split_keeping_close_balance(df, left_split_frac, disregard_subject_ids=False):
    """
    Split df into two pieces, holding to approximate left_split_frac but only approximate in the
    case that single subject ids should not be split up (disregard_subject_ids=False). 
    Will drop samples in order to attain similar class imabalance (plus or minus one sample) 
    """
    df_balance = parse_classes(df)['Balance']
    if disregard_subject_ids:
        left_split, right_split = basic_stratified_split(df=df, left_split_frac=left_split_frac)    
    else:
        left_split, right_split = approx_optimal_by_subject_split(df=df, left_split_frac=left_split_frac)
        # drop samples to restore similar class balance
        left_split, left_dropped_dict = rebalance(df=left_split, target_balance=df_balance)
        right_split, right_dropped_dict = rebalance(df=right_split, target_balance=df_balance)

        print()
        print(f"\nTo keep balance similar after split by subject ID, we dropped this many samples")
        print(f"From left split: Class 0: {len(left_dropped_dict[0])}, Class 1: {len(left_dropped_dict[1])}")
        print(f"From right split: Class 0: {len(right_dropped_dict[0])}, Class 1: {len(right_dropped_dict[1])}\n")
    
    # Now shuffle and reset indices
    left_split = left_split.sample(frac=1).reset_index(drop=True)
    right_split = right_split.sample(frac=1).reset_index(drop=True)

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
    

    # We shuffle here and many other places in this script
    orig_train = orig_train.sample(frac=1).reset_index(drop=True)
    orig_val = orig_val.sample(frac=1).reset_index(drop=True)
    orig_test = orig_test.sample(frac=1).reset_index(drop=True)

    # We are hoping to maintain similar class imbalance across all output sets 
    orig_train_balance = parse_classes(orig_train)['Balance']
    orig_val_balance = parse_classes(orig_val)['Balance']
    orig_test_balance = parse_classes(orig_test)['Balance']

    # print out some info
    print(f"\nOriginal train had a count of {len(orig_train)} with a balance of {orig_train_balance}")
    print(f"Original val had a count of {len(orig_val)} with a balance of {orig_val_balance}")
    print(f"Original test had a count of {len(orig_test)} with a balance of {orig_test_balance}\n")
    
    # We assume incoming sets already have similar imbalance
    tol = 0.001
    if np.absolute(orig_train_balance - orig_val_balance) > tol:
        raise ValueError(f"Original train and val balance are off by more than the set hard coded tolerance of: {tol}")
    if np.absolute(orig_train_balance - orig_test_balance) > tol:
        raise ValueError(f"Original train and test balance are off by more than the set hard coded tolerance of: {tol}")
   
    # infer classes
    classes = list(orig_train['valuetopredict'].unique())

    # NOTE: For our current use case we assume binary classification, this effects class balance computations
    # which exist throughout the code (ie, only balance of first two classes is governed). 
    # We also assume labels are 0 and 1 (we explicitly refer to each class as we want to control
    # The class balance ratio to be class0/total).
    if len(classes) != 2:
        raise ValueError("Current use case assumes binary classification (see note in code next to exception raising)")
    if set(classes) != set([0, 1]):
        raise ValueError("Assumption is binary labels 0 and 1.(see note in code next to exception raising)")

    print(f"\nDetecting the complete class list from the train csv and found: {classes}.\n")

    # now split test to create PM_test and PM_pop (PM_ref will be created using PM_pop)
    PM_pop, PM_test = split_keeping_close_balance(orig_test, orig_test_portion_to_pop)

    print(f"\nPM_pop is now set with a size of: {len(PM_pop)}")
    print(f"PM_test is currently size: {len(PM_test)} but may need to change to account for training set size.\n")

    
    # PM_test and PM_train need to be of equal size
    if len(PM_test) > len(orig_train):
        if not allow_dropped_orig_test_samples:
            raise ValueError(f"Not allowing to drop test samples but we do not have enough training samples to match fraction of samples going to PM_test from orig_test.")
        else:
            PM_test, _ = split_keeping_close_balance(PM_test,left_split_frac=len(orig_train)/float(len(PM_test)))
            PM_train = orig_train
    else:
        if not allow_dropped_orig_train_samples:
            raise ValueError(f"Not allowing to drop train samples but we do not have enough testing samples after pulling off samples for PM_other.")
        else:
            PM_train, samples_split_from_orig_train = split_keeping_close_balance(orig_train, left_split_frac=len(PM_test)/float(len(orig_train)))   

    # PM_pop, samples split from orig_train, and orig_val combine to create PM_ref (maybe slightly off in balance after combining) 
    
    # This is a long accounting to ensure we carefully add samples to form PM_ref in priority order
    # (PM_pop are best, followed by orig_val then samples_split_from_orig_train)
    if len(orig_train) < len(PM_pop):
        unbal_PM_ref, _ = basic_stratified_split(df=PM_pop, 
                                                 left_split_frac=len(orig_train)/float(len(PM_pop)))
    elif len(orig_train) == len(PM_pop):
        unbal_PM_ref = PM_pop
    elif len(orig_train) < len(PM_pop) + len(orig_val):
        num_needed_from_orig_val = len(orig_train) - len(PM_pop)
        samples_split_from_orig_val, _ = basic_stratified_split(df=orig_val, 
                                                             left_split_frac=num_needed_from_orig_val/float(len(orig_val)))
        unbal_PM_ref = pd.concat([PM_pop, samples_split_from_orig_val])
    elif len(orig_train) == len(PM_pop) + len(orig_val): 
        unbal_PM_ref = pd.concat([PM_pop, orig_val])
    elif len(orig_train) < len(PM_pop) + len(orig_val) + len(samples_split_from_orig_train):
        num_needed_from_split_train_samples = len(orig_train) - len(PM_pop) - len(orig_val)
        further_split_from_orig_train, _ = basic_stratified_split(df=samples_split_from_orig_train,
                                                                  left_split_frac=num_needed_from_split_train_samples/float(len(samples_split_from_orig_train)))
        unbal_PM_ref = pd.concat([PM_pop, orig_val, further_split_from_orig_train])
    elif len(orig_train) >= len(PM_pop) + len(orig_val) + len(samples_split_from_orig_train):
        unbal_PM_ref = pd.concat([PM_pop, orig_val, samples_split_from_orig_train])
 
    unbal_PM_ref_balance = parse_classes(unbal_PM_ref)['Balance']
    
    # Now rebalance PM_ref to get close to the balance of orig_train (shouldn't be off by too much)
    PM_ref, samples_dropped_from_unbal_PM_ref = rebalance(unbal_PM_ref, target_balance=orig_train_balance)
    output_dict = {key: len(value) for (key, value) in samples_dropped_from_unbal_PM_ref.items()}
    print(f"\nSample counts dropped from unbal_PM_ref to go from balance of: {unbal_PM_ref_balance} to: {orig_train_balance} were: {output_dict}")

    # because of the subject splits, PM_test and PM_train may not be the same length
    if len(PM_test) > len(PM_train):
        PM_test, _ = basic_stratified_split(df=PM_test, left_split_frac=len(PM_train)/float(len(PM_test)))
    elif len(PM_test) > len(PM_train):
        PM_train, _ = basic_stratified_split(df=PM_train, left_split_frac=len(PM_test)/float(len(PM_train)))
   
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

    final_pm_train_bal = parse_classes(PM_train)['Balance']
    final_pm_test_bal = parse_classes(PM_test)['Balance']
    final_pm_pop_bal = parse_classes(PM_pop)['Balance']
    final_pm_ref_bal = parse_classes(PM_ref)['Balance']

    print(f"\nTrain, Test, Pop, Ref Counts: {final_pm_train_length, final_pm_test_length, final_pm_pop_length, final_pm_ref_length}")
    print(f"Train, Test, Pop, Ref Bals: {final_pm_train_bal, final_pm_test_bal, final_pm_pop_bal, final_pm_ref_bal}\n")
    
    # PM_test and PM_train need to be the same length
    if final_pm_train_length != final_pm_test_length:
        raise ValueError("Final PM_train len: {final_pm_train_length} is not equal to final PM_test len: {final_pm_test_length}")
    # PM_train, PM_test, PM_pop, and PM_ref need to have similar class balances
    smallest_set_size = min(final_pm_train_length, final_pm_pop_length, final_pm_ref_length)
    tolerance = 2/float(smallest_set_size)
    if abs(final_pm_train_bal - final_pm_test_bal) > tolerance:
        raise ValueError(f"Final balances of train {final_pm_train_bal} and test {final_pm_test_bal} are not within the tolerance of {tolerance}")
    if abs(final_pm_train_bal - final_pm_pop_bal) > tolerance:
        raise ValueError(f"Final balances of train {final_pm_train_bal} and pop {final_pm_pop_bal} are not within the tolerance of {tolerance}")
    if abs(final_pm_train_bal - final_pm_ref_bal) > tolerance:
        raise ValueError(f"Final balances of train {final_pm_train_bal} and ref {final_pm_ref_bal} are not within the tolerance of {tolerance}")
    
    
    # PM_ref needs to have similar class balance to orig_train
    if abs(final_pm_ref_bal-orig_train_balance) > tolerance:
        raise ValueError(f"Final balances of ref {final_pm_ref_bal} and original train {orig_train_balance} are not within the tolerance of {tolerance}")
    
    PM_train_csv_path = os.path.join(new_csv_folder, data_name + "_pm_train_w_orig_test_balance.csv") 
    PM_test_csv_path = os.path.join(new_csv_folder, data_name + "_pm_test_w_orig_test_balance.csv") 
    PM_pop_csv_path = os.path.join(new_csv_folder, data_name + "_pm_pop_w_orig_test_balance.csv")
    PM_ref_csv_path = os.path.join(new_csv_folder, data_name + "_pm_ref_w_orig_train_balance.csv")

    # now shuffle the resulting dataframes once more
    PM_train = PM_train.sample(frac=1).reset_index(drop=True)
    PM_test = PM_test.sample(frac=1).reset_index(drop=True)
    PM_pop = PM_pop.sample(frac=1).reset_index(drop=True)
    PM_ref = PM_ref.sample(frac=1).reset_index(drop=True)

    PM_train.to_csv(PM_train_csv_path, index=False)
    PM_test.to_csv(PM_test_csv_path, index=False)
    PM_pop.to_csv(PM_pop_csv_path, index=False)
    PM_ref.to_csv(PM_ref_csv_path, index=False)
