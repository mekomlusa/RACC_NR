'''
A helper script to scrape and format the individual results from nested folders.

Written by RL for the RACC_NR project.

12/8/2018
'''

import os
import pandas as pd
import argparse

result_df = pd.DataFrame(columns=['training_type','testing_type','file_type','joint_matrix','mariginal_prediction','marginal_labels','mutual_information'])

# function to save the dataframe
def output_file(df, filen, output_path):
    outputname = filen+".csv"
    if len(df) > 0:
        df.to_csv(output_path+outputname,sep=',', na_rep=" ", encoding='utf-8', index_label=False, index=False) 
        print("Length of the dataframe:"+str(len(df)))
        print("Results have been saved under "+output_path+" as "+filen)

parser = argparse.ArgumentParser(
        description='RACC_NR U-Net data aggregation script')
parser.add_argument('--p', required=True,
                    metavar="/path/to/dataset/",
                    help='Path to the saved structure folder')
args = parser.parse_args()
dir_path = args.p

# force the path to end with /
if dir_path[-1] != '/':
    dir_path += '/'

# full walking on the folder
for root, dirs, files in os.walk(dir_path):
    for file_ in files:
        full_path = os.path.join(root, file_)
        full_path_elem = full_path.split('/')
        if 'test_summary.txt' in full_path_elem[-1]:
            # found the summary file
            with open(full_path, 'r') as f:
                data = f.read()
                data_split = data.split('\n')
                df = pd.DataFrame([full_path_elem[-4],full_path_elem[-3],full_path_elem[-2],data_split[1]+'\n'+data_split[2],data_split[4],data_split[6],data_split[-1]],index=[0])
                result_df = pd.concat([result_df, df], ignore_index=True)
                
# once finish, output the dataframe            
output_file(result_df, 'RACC_NR_result',dir_path)