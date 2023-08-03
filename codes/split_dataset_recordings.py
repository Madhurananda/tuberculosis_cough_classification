#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:12:22 2020

@author: madhu
"""

"""
This script reads all the feature data and then splits the patinets lists in train, test
and validation set. 
This script generates splits which are used in LR clasifier and other experiments. 
"""


import os
import random
import numpy as np
import pandas as pd

from glob import glob
from natsort import natsorted
from sklearn.model_selection import StratifiedKFold, train_test_split
# from helper import *
import config


def reserve_set(recs = [],labels = [], res_size = None):

    """
    Reserve the data from recordings
    in recs in a separate dataset.

    Choose to reserve a list of recordings, 
    or a percentage of df by passing either
    recs or res_size

    """

    # Convert to array for broadcasting
    labels = np.array(labels)
    recs = np.array(recs)

    # Number of recordings in big set
    N = len(recs)
    # Number of recordings in reserved set
    N_res = int(res_size * N)
    
    # while (res_labels.count(0) == N_res/2 and res_labels.count(1) == N_res/2):
    
    # # Select N_res random recordings
    #     rand_idx = random.sample(range(N),N_res)
    #     res_labels = list(labels[rand_idx])
    
    # # if res_labels.count(0) == N_res/2 and res_labels.count(1) == N_res/2:
    #     res_recs = recs[rand_idx]
    
    condition = 1
    
    # Loop until a balanced reserved set it found
    while condition == 1:
    # while !(res_labels.count(0) == N_res/2 and res_labels.count(1) == N_res/2):
        # Select N_res random recordings
        
        rand_idx = random.sample(range(N),N_res)
        res_labels = list(labels[rand_idx])
        
        # print('random no.', rand_idx)
        
        """
        New Method
        """
        # At least 2 recordings of each classes
        if res_labels.count(0) > 1 and res_labels.count(1) > 1:
            res_recs = recs[rand_idx]
            condition = 0
            # break	
        
        """
        Old Method
        """
        # # Both classes present in reserved set
        # print('N ', N)
        # print('N_res', N_res)
        # print(res_labels.count(0))
        # print(res_labels.count(1))
        
        # if res_labels.count(0) == int(N_res/2) and res_labels.count(1) == int(N_res/2):
        #     res_recs = recs[rand_idx]
        #     condition = 0
            # break
        
    res_recs = list(res_recs)
    recs = [r for r in recs if r not in res_recs]
    
    return recs, res_recs


def save_sets(splits_dir,train_recs, test_recs, val_recs, k, f):

    """
    Save the three sets to the train, test and val
    directories
    """
    fout = "".join([splits_dir, 'val_size=', str(val_size), '/', str(k), '/', os.path.basename(f)[:-4],".txt"])

    train = "".join(["train=",repr(train_recs),'\n'])
    test = "".join(["test=",repr(test_recs),'\n'])
    val = "".join(["val=",repr(val_recs),'\n'])
    
    if os.path.isdir( "".join([ splits_dir, 'val_size=', str(val_size), '/' ])) == False:
        # print('The directory exists.')
    #     print('')
    # else:
        os.mkdir("".join([ splits_dir, 'val_size=', str(val_size), '/' ]))
    
    if os.path.isdir( "".join([ splits_dir, 'val_size=', str(val_size), '/', str(k), '/' ])) == False:
        # print('The directory exists.')
    #     print('')
    # else:
        os.mkdir("".join([ splits_dir, 'val_size=', str(val_size), '/', str(k), '/' ]))
    
    out = open(fout,'w')
    out.write(train)
    out.write(test)
    out.write(val)
    out.close()


def save_sets_no_val(splits_dir,train_recs, test_recs, k, f):

    """
    Save the three sets to the train, test and val
    directories
    """
    fout = "".join([splits_dir, 'no_val', '/', str(k), '/', os.path.basename(f)[:-4],".txt"])

    train = "".join(["train=",repr(train_recs),'\n'])
    test = "".join(["test=",repr(test_recs),'\n'])
    # val = "".join(["val=",repr(val_recs),'\n'])
    
    if os.path.isdir( "".join([ splits_dir, 'no_val', '/' ])) == False:
        # print('The directory exists.')
    #     print('')
    # else:
        os.mkdir("".join([ splits_dir, 'no_val', '/' ]))
    
    if os.path.isdir( "".join([ splits_dir, 'no_val', '/', str(k), '/' ])) == False:
        # print('The directory exists.')
    #     print('')
    # else:
        os.mkdir("".join([ splits_dir, 'no_val', '/', str(k), '/' ]))
    
    out = open(fout,'w')
    out.write(train)
    out.write(test)
    # out.write(val)
    out.close()


def hasIntersection(a,b):
    return set(a).isdisjoint(b)


def test_splits(f, splits_dir):
    
    """
    Go through directories 1..N_FOLD, 
    load the file with rec names
    print out dev and test sets
    make sure there aren't any repetitions
    """

    # print 'Testing splits...',

    for k in range(1,N_FOLD):

        f_split = "".join([splits_dir, 'val_size=', str(val_size), '/', str(k),'/',os.path.basename(f)[:-4],'.txt'])

        data = np.loadtxt(f_split,delimiter='=',dtype=str)

        train = eval(data[0,1])
        test = eval(data[1,1])
        val = eval(data[2,1])

        # if all three sets are disjoint, continue
        if hasIntersection(train, test) and hasIntersection(train,val) and hasIntersection(val,test):
        # if hasIntersection(train, test):
            continue
        
        else:
#           print (k,'has intersection')
            exit()

def test_splits_no_val(f, splits_dir):
    
    """
    Go through directories 1..N_FOLD, 
    load the file with rec names
    print out dev and test sets
    make sure there aren't any repetitions
    """

    # print 'Testing splits...',

    for k in range(1,N_FOLD):

        f_split = "".join([splits_dir, 'no_val', '/', str(k),'/',os.path.basename(f)[:-4],'.txt'])

        data = np.loadtxt(f_split,delimiter='=',dtype=str)

        train = eval(data[0,1])
        test = eval(data[1,1])
        # val = eval(data[2,1])

        # if all three sets are disjoint, continue
#       if hasIntersection(train, test) and hasIntersection(train,val) and hasIntersection(val,test):
        if hasIntersection(train, test):
            continue
        
        else:
#           print (k,'has intersection')
            exit()



def setup_SKF(feat_files):
	"""
	Load the first feature file
	Get all unique recording names
	and their labels

	Make SKF from labels
	"""

	# Load the data
	df = pd.read_csv(feat_files[0], sep=';')

	# Get the list of recordings and labels
	recs = np.array(df.Study_Num.unique())
	
	labels = np.array([df[df.Study_Num == rec].TB_status.values[0] for rec in recs])

	# Stratified Splitter
	skf = StratifiedKFold(n_splits = N_FOLD)

	return recs, labels, skf

def split_dataset(splits_dir, feat_files):

    # Setup the Stratified K Fold
    recs, labels, skf = setup_SKF(feat_files)

    for feat_file in feat_files:
        
        print ("Splitting for",os.path.basename(feat_file))

        k = 1
        
        # print('It came here 1')
        
        for dev_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
#       for dev_idx, test_idx in skf:
            # print('value of k is: ', k)
            dev_recs = recs[dev_idx]
            # print('It came here 2')
            dev_labels = labels[dev_idx]
            # print(dev_labels)
            test_recs = list(recs[test_idx])
            # print(test_recs)
            # Split dev set into train and val sets
            train_recs, val_recs = reserve_set(dev_recs,dev_labels,res_size=val_size)
            # print('It came here 3')
            # Save the sets to disk
            save_sets(splits_dir,train_recs, test_recs, val_recs, k, feat_file)
            
            k += 1
            # print('It came here 4')
        
        # print('It came here 5')
        # Test that splits were done correctly, exit if sets overlap
        test_splits(feat_file, splits_dir)

def split_dataset_no_val(splits_dir, feat_files):

    # Setup the Stratified K Fold
    recs, labels, skf = setup_SKF(feat_files)

    for feat_file in feat_files:
        
        print ("Splitting for",os.path.basename(feat_file))

        k = 1
        
        # print('It came here 1')
        
        for dev_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
#       for dev_idx, test_idx in skf:
            # print('value of k is: ', k)
            train_recs = list(recs[dev_idx])
            # print('It came here 2')
            # dev_labels = labels[dev_idx]
            # print(dev_labels)
            test_recs = list(recs[test_idx])
            # print(test_recs)
            # Split dev set into train and val sets
            # train_recs, val_recs = reserve_set(dev_recs,dev_labels,res_size=val_size)
            # print('It came here 3')
            # Save the sets to disk
            save_sets_no_val(splits_dir,train_recs, test_recs, k, feat_file)
            
            k += 1
            # print('It came here 4')
        
        # print('It came here 5')
        # Test that splits were done correctly, exit if sets overlap
        test_splits_no_val(feat_file, splits_dir)


val_sizes = config.val_sizes_split

N_FOLD = config.N_FOLDS

splits_dir = config.splits_dir

feat_list = config.feat_list

for val_size in val_sizes: 
    if val_size == 0:
        split_dataset_no_val(splits_dir, natsorted(feat_list))
    else:
        split_dataset(splits_dir, natsorted(feat_list))


