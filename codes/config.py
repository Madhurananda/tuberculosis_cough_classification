#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:00:11 2020

@author: madhu
"""

from glob import glob


# Split dataset
val_sizes_split = [0, 0.1, 0.15, 0.2, 0.25]

# CNN Classifier
splits_dir_CNN = '../data/CNN_classifier/splits/'
feat_list_CNN = glob('../data/CNN_classifier/features_dataset/*')

# LR Classifier 
val_sizes_LRc = [0.1, 0.15, 0.2, 0.25]
splits_dir = '../data/feature_data/splits/'
feat_list = glob('../data/feature_data/features_dataset/*')

GAMMA = 0.5
CV_SCORER = 'AUC'
N_FOLDS = 15


# Number of bootstrap samples
B = 100
splits_dir_bootstrap = '../data/LR_classifier/splits/no_val/'
features_list = glob('../data/feature_dataset/best_reduced_LR/*')
