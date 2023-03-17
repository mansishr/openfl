import argparse, os

from privacy_meter_constructCSVs_watch_subject_IDs import construct_PM_csvs

# hard coded choices
orig_test_portion_to_pop = 0.5  
                      

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--orig_csv_pardir',
        type=str,
        help='Absolute path to the folder holding the original train, val, and test csv files')
    argparser.add_argument(
        '--PM_csv_pardir',
        type=str,
        help='Absolute path to the folder that will hold the PM csvs')
    argparser.add_argument(
        '--data_name',
        type=str,
        default='PM_tutorial',
        help='Tag used to name the output files')
    argparser.add_argument(
        '--orig_test_portion_to_pop',
        type=float,
        default=0.5,
        help='How much or the original test data to put intot the PM population set.')
   
    
    args = argparser.parse_args()

    kwargs = vars(args)

    orig_csv_pardir = args.orig_csv_pardir
    PM_csv_pardir = args.PM_csv_pardir
    data_name = args.data_name

    inst_names = os.listdir(orig_csv_pardir)
    

    for inst_name in inst_names:
        
        orig_train_csv_path = os.path.join(orig_csv_pardir, inst_name, inst_name + '_train.csv')
        orig_val_csv_path = os.path.join(orig_csv_pardir, inst_name, inst_name + '_val.csv')
        orig_test_csv_path = os.path.join(orig_csv_pardir, inst_name, inst_name + '_test.csv')

        new_csv_folder = os.path.join(PM_csv_pardir, inst_name)



    construct_PM_csvs(orig_train_csv_path, 
                      orig_val_csv_path, 
                      orig_test_csv_path, 
                      new_csv_folder, 
                      data_name, 
                      orig_test_portion_to_pop)



