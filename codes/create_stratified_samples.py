#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:18:01 2020

@author: madhu
"""

import os
import pandas as pd
import numpy as np
import helper_stratified as help

from glob import glob
from natsort import natsorted

import config

"""
Make stratified splits for
each fold specifically for this
test_csv

This is necessary to compare the
results over multiple iterations because
each time stratified_sample is called, we might
get a different subsampled set depending on the
class distribution - we need a stratified sample

Stratified sampling is done within each fold
and using the per-recording labels.
So n number of recording are sampled with
replacement for every b iteration and
stratification is ensured.

"""

# Number of bootstrap samples
B = config.B
N_FOLD = config.N_FOLDS
splits_dir = config.splits_dir_bootstrap
# test_csv_list = config.features_list



classifier_idx = 1 # LR
# classifier_idx = 2 # KNN
# classifier_idx = 3 # SVM
# classifier_idx = 4 # MLP

if classifier_idx == 1:
    classifier_name = 'LR'

elif classifier_idx == 2:
    classifier_name = 'KNN'

elif classifier_idx == 3:
    classifier_name = 'SVM'

elif classifier_idx == 4:
    classifier_name = 'MLP'


proj_database = '../data/'

# feat_dir = proj_database + 'feature_data/features_dataset/'
result_dir = proj_database + 'results/' + classifier_name+ '_classifier/'

feat_dir = result_dir + 'best_reduced_feat/'

output_File = result_dir + 'text_outputs/class_SFS_output.txt'
splits_dir = proj_database + 'feature_data/splits/no_val/'

test_csv_list = natsorted(os.listdir(feat_dir))


for test_csv in test_csv_list:

    print ('\n',os.path.basename(test_csv))
    
    # Skip MFCC files...
    # if 'MFCC' in test_csv:
    #     continue

    # Number of fbanks
    # N = test_csv.split('.')[-2].split('=')[-1]
    N = test_csv.split('=')[1].split('_')[0]
    
    """ Setup output directory"""
    bootstrap_out_dir = '../data/feature_dataset/bootstrap_samples/N={}/B={}/'.format(N, B)
    if not os.path.isdir(bootstrap_out_dir):
        os.makedirs(bootstrap_out_dir)

    # Read in the feature data
    df = pd.read_csv(test_csv, sep=';').dropna(thresh=2)
    
    for k in range(1, N_FOLD+1):
        print ('Fold: ', k)
        
        # Split the data into train and test
        train_df, test_df = help.load_splits(splits_dir, df, test_csv, k)
        
        # Get the test patient names and their labels
        test_recs = np.unique(test_df.Study_Num)
        test_labels = [0 if 'NTB' in r else 1 for r in test_recs]
        
        # print('Length of test labels: ', len(test_labels))
        
        # Make a temp DF with recordings and labels to use for sampling
        temp_df = pd.DataFrame(np.column_stack((test_recs,test_labels)), columns=['Study_Num', 'TB_status'])
        
        # Output filename
        f_out = bootstrap_out_dir + 'FOLD_{}.csv'.format(k)
        
        out = open(f_out, 'w')
        for b in range(B):
            # Sample until both classes present
            while True:
                # df_ = temp_df.sample(n=int(len(temp_df.index)/2), replace=True)
                if len(temp_df.index) > 3:
                    # Take a sample with n = half of length of temp_df
                    df_ = temp_df.sample(n=int(len(temp_df.index)/2), replace=True)
                else:
                    df_ = temp_df.sample(n=len(temp_df.index), replace=True)
                
                # Check if both classes are present
                if list(df_.TB_status.values).count(0) >= 1 and list(df_.TB_status.values).count(1) >= 1:
                    break

            out.write('{};{}\n'.format(b, repr(list(df_.Study_Num.values))))
        out.close()
    
    
    print ('')
    
    
    
    
    