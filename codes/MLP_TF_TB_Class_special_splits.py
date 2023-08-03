#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:12:22 2020

@author: madhu
"""

"""
This script runs the LR classifier for every features and outputs results like:
table 6.3 in Renier's thesis. 
This scripts calculates ROC curve for both TIS and ADS for a list of features 
found inside a directory and even for individually found best features. 
"""

import os
import csv
import sys
import time
import random
import datetime
import numpy as np
import pandas as pd
import pickle
from natsort import natsorted
from numpy import interp

from sklearn.metrics import *
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.preprocessing import binarize, scale, robust_scale
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, PredefinedSplit, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD, Adam
from keras.constraints import maxnorm

sys.path.insert(0, './Renier_scripts')
from helper import *
import config

# This function writes content to a file. 
def write_file(file_name, to_write):
    # Append-adds at last 
    file = open(file_name,"a")#append mode 
    file.write(to_write) 
    file.close() 

def ROC_analysis(prob_ref_list, thresh_=True, return_rates=False, plot_=False, fname=None, name=''):
    """
    :param prob_ref_list List of tuples (prob, ref_label)
    :param plot:    Flag for plotting ROC curve
    :param thresh_  Flag to return equal error threshold
    :param fname    Name of feature file - make dir with that name
    :param name     Name to be put in graph title. Name of eval method
    :return:        optimized predictions (opt_pred)
                    area under the curve (AUC_ACC)
    """

    # prob_arr = np.array(prob_ref_list)
    # probs = prob_arr[:, 0]
    # y_true = prob_arr[:, 1]
    
    [probs, y_true] = zip(*prob_ref_list)
    fpr, tpr, thresholds = roc_curve(y_true, probs, drop_intermediate=False)
    auc_acc = auc(fpr, tpr)

    i = np.arange(len(tpr))
    roc = pd.DataFrame({'fpr': pd.Series(fpr, index=i),
                        'tpr': pd.Series(tpr, index=i),
                        '1-fpr': pd.Series(1 - fpr, index=i),
                        'tf': pd.Series(tpr - (1 - fpr), index=i),
                        'thresholds': pd.Series(thresholds, index=i)
                        })
    idx = (roc.tf).abs().argmin()
    threshold = roc.thresholds.iloc[idx]

    # Do diagnosis with optimized threshold
    opt_preds = map(int, binarize(np.array(probs).reshape(1, -1), threshold=threshold)[0])

    if plot_:
        fname = os.path.basename(fname)[:-4]
        fig_out_dir = '{}/{}'.format(ROC_overall_figs_dir, fname)
        if not os.path.isdir(fig_out_dir):
            os.mkdir(fig_out_dir)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = {})'.format(auc_acc))
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for {}'.format(name))
        plt.savefig(fig_out_dir + '/ROC_{}.png'.format(name))
        plt.close()

    """
    Spagetti code but meh
    """
    if return_rates:

        if thresh_:
            return opt_preds, auc_acc, threshold, tpr, fpr

        else:
            return opt_preds, auc_acc, tpr, fpr

    else:
        if thresh_:
            return opt_preds, auc_acc, threshold

        else:
            return opt_preds, auc_acc


# def load_splits(full_df, f, k):
#     """
# 	full_df:	Complete dataframe
# 	f:			Name of feature dataset (csv)
# 	k:			Split number
# 	"""
#     # f_split = "".join([splits_dir, str(k), '/', os.path.basename(f)[:-4], ".txt"])
#     f_split = splits_dir + 'new_splits.txt'
    
#     splits_data = np.loadtxt(f_split, delimiter='\n', dtype=str)
#     # splits_data = np.loadtxt(f_split, delimiter='=', dtype=str)

#     train_recs = eval(splits_data[0, 1])
#     test_recs = eval(splits_data[1, 1])
#     val_recs = eval(splits_data[2, 1])
    
#     train_df = full_df[full_df.Study_Num.isin(train_recs)]
#     test_df = full_df[full_df.Study_Num.isin(test_recs)]
#     val_df = full_df[full_df.Study_Num.isin(val_recs)]
    
#     return train_df, test_df, val_df


# def cv_param_estimation(df):
#     def make_ps(df, n_folds=4):
#         """
#         :param df: full dataframe
#         :param n_folds: number of folds

#         :return: list indicating which samples
#                  belong to which fold
#         """

#         # Get the list of recordings and labels
#         recs = np.array(df.Study_Num.unique())

#         labels = np.array([df[df.Study_Num == rec].TB_status.values[0] for rec in recs])

#         # skf = StratifiedKFold(labels, n_folds=n_folds)
#         skf = StratifiedKFold(n_splits=n_folds)

#         y_ps = np.zeros(shape=(len(df.index),))
        
#         fold = 0
        
#         for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        
#         # for train_idx, test_idx in skf:
#             test_recs = recs[test_idx]

#             ps_test_idx = np.array(df[df['Study_Num'].isin(test_recs)].index)

#             y_ps[ps_test_idx] = fold

#             fold += 1
        
#         return PredefinedSplit(y_ps)

#     def run_gridsearch(X, y, Cs, cv, pen_ratios):
#         """
#         Tailoring specifically for LRCV

#         When using a different classifier, will need to use
#         GridSearchCV method and this process would
#         be different.
#         """

#         if CV_SCORER == 'KAPPA':
#             kappa_scorer = make_scorer(cohen_kappa_score)
#             grid_search = LogisticRegressionCV(Cs=Cs, cv=cv, scoring=kappa_scorer)

#         elif CV_SCORER == 'AUC':
#             # grid_search = LogisticRegressionCV(Cs=Cs, cv=cv, scoring='roc_auc')
            
#             grid_search = LogisticRegressionCV(Cs=Cs, cv=cv, scoring='roc_auc', max_iter = 100000000, solver='saga', l1_ratios = pen_ratios, penalty='elasticnet', n_jobs = -1)
#             # grid_search = LogisticRegressionCV(Cs=Cs, cv=cv, scoring='roc_auc', max_iter = 100000000, solver='saga', l1_ratios = pen_ratios, penalty='elasticnet', class_weight='balanced', n_jobs = -1)
            
#         else:
#             grid_search = LogisticRegressionCV(Cs=Cs, cv=cv)
        
#         # Scale the data
#         # X_scaled = scale(X)
        
#         X_r_scaled = robust_scale(X)
        
#         # Fit GS with this data
#         grid_search.fit(X_r_scaled, y)
        
#         gs_opt_params_C = {'C': grid_search.C_[0]}
        
#         gs_opt_params_l1_ratio = {'l1_ratio': grid_search.l1_ratio_[0]}
        
#         # gs_opt_params = grid_search.best_params_
        
#         print (gs_opt_params_C)
#         print(gs_opt_params_l1_ratio)
        
#         '''
#         LogisticRegressionCV.scores_ gives the score for all the folds.
#         GridSearchCV.best_score_ gives the best mean score over all the folds.
#         '''
        
#         print ('Local Max auc_roc:', grid_search.scores_[1].max())  # is wrong
#         print ('Max auc_roc:', grid_search.scores_[1].mean(axis=0).max())  # is correct
        
#         to_save = 'Max auc_roc:' +  str(grid_search.scores_[1].mean(axis=0).max()) + '\n'
        
#         to_save += str(gs_opt_params_C) + '\n' + str(gs_opt_params_l1_ratio)
#         # to_save = str(gs_opt_params_C) + '\n' + str(gs_opt_params_l1_ratio)

#         return gs_opt_params_C, gs_opt_params_l1_ratio, to_save
    
#     df_ = df.copy()
#     # Reset index for referencing
#     df_.reset_index(inplace=True)
    
#     # Make the PredefinedSplit label
#     # ps = make_ps(df_, n_folds=2)
#     ps = make_ps(df_)
    
#     # labels
#     y_val = df_.TB_status.values
#     # data
#     X_val = df_[feat_names]
    
#     LR_model = LogisticRegression(max_iter=100000000, solver='saga', penalty='elasticnet', n_jobs = -1)
#     # LR_model = LogisticRegression(max_iter=100000000, solver='saga', penalty='elasticnet', class_weight='balanced', n_jobs = -1)
    
#     Cs = np.logspace(-5, 5, 10)
#     pen_ratios = np.linspace(0, 1, num=6)
    
#     # Cs = np.logspace(-7, 7, 30)
#     # pen_ratios = np.linspace(0, 1, num=21)
    
    
#     LR_params_C, LR_params_l1_ratio, to_save = run_gridsearch(X_val, y_val, Cs, cv=ps, pen_ratios= pen_ratios)
    
#     LR_model.set_params(**LR_params_C)
#     LR_model.set_params(**LR_params_l1_ratio)

#     return LR_model, to_save







def cv_param_estimation_LR(df):
    def make_ps(df, n_folds=4):
        """
        :param df: full dataframe
        :param n_folds: number of folds

        :return: list indicating which samples
                 belong to which fold
        """

        # Get the list of recordings and labels
        recs = np.array(df.Study_Num.unique())

        labels = np.array([df[df.Study_Num == rec].TB_status.values[0] for rec in recs])

        # skf = StratifiedKFold(labels, n_folds=n_folds)
        skf = StratifiedKFold(n_splits=n_folds)

        y_ps = np.zeros(shape=(len(df.index),))
        
        fold = 0
        
        for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        
        # for train_idx, test_idx in skf:
            test_recs = recs[test_idx]

            ps_test_idx = np.array(df[df['Study_Num'].isin(test_recs)].index)

            y_ps[ps_test_idx] = fold

            fold += 1
        
        return PredefinedSplit(y_ps)

    def run_gridsearch(X, y, Cs, cv, pen_ratios):
        """
        Tailoring specifically for LRCV

        When using a different classifier, will need to use
        GridSearchCV method and this process would
        be different.
        """

        if CV_SCORER == 'KAPPA':
            kappa_scorer = make_scorer(cohen_kappa_score)
            grid_search = LogisticRegressionCV(Cs=Cs, cv=cv, scoring=kappa_scorer)

        elif CV_SCORER == 'AUC':
            # grid_search = LogisticRegressionCV(Cs=Cs, cv=cv, scoring='roc_auc')
            
            grid_search = LogisticRegressionCV(Cs=Cs, cv=cv, scoring='roc_auc', max_iter = 100000000, solver='saga', l1_ratios = pen_ratios, penalty='elasticnet', n_jobs = 20)
            # grid_search = LogisticRegressionCV(Cs=Cs, cv=cv, scoring='roc_auc', max_iter = 100000000, solver='saga', l1_ratios = pen_ratios, penalty='elasticnet', class_weight='balanced', n_jobs = 20)
            
        else:
            grid_search = LogisticRegressionCV(Cs=Cs, cv=cv)
        
        # Scale the data
        # X_scaled = scale(X)
        
        X_r_scaled = robust_scale(X)
        
        # Fit GS with this data
        grid_search.fit(X_r_scaled, y)
        
        gs_opt_params_C = {'C': grid_search.C_[0]}
        
        gs_opt_params_l1_ratio = {'l1_ratio': grid_search.l1_ratio_[0]}
        
        # gs_opt_params = grid_search.best_params_
        
        print (gs_opt_params_C)
        print(gs_opt_params_l1_ratio)
        
        '''
        LogisticRegressionCV.scores_ gives the score for all the folds.
        GridSearchCV.best_score_ gives the best mean score over all the folds.
        '''
        
        print ('Local Max auc_roc:', grid_search.scores_[1].max())  # is wrong
        print ('Max auc_roc:', grid_search.scores_[1].mean(axis=0).max())  # is correct
        
        to_save = 'Max auc_roc:' +  str(grid_search.scores_[1].mean(axis=0).max()) + '\n'
        
        to_save += str(gs_opt_params_C) + '\n' + str(gs_opt_params_l1_ratio)
        # to_save = str(gs_opt_params_C) + '\n' + str(gs_opt_params_l1_ratio)

        return gs_opt_params_C, gs_opt_params_l1_ratio, to_save
    
    df_ = df.copy()
    # Reset index for referencing
    df_.reset_index(inplace=True)
    
    # Make the PredefinedSplit label
    # ps = make_ps(df_, n_folds=2)
    ps = make_ps(df_)
    
    # labels
    y_val = df_.TB_status.values
    # data
    X_val = df_[feat_names]
    
    LR_model = LogisticRegression(max_iter=100000000, solver='saga', penalty='elasticnet', n_jobs = 20)
    # LR_model = LogisticRegression(max_iter=100000000, solver='saga', penalty='elasticnet', class_weight='balanced', n_jobs = 20)
    
    Cs = np.logspace(-5, 5, 10)
    pen_ratios = np.linspace(0, 1, num=6)
    
    # Cs = np.logspace(-7, 7, 30)
    # pen_ratios = np.linspace(0, 1, num=21)
    
    
    LR_params_C, LR_params_l1_ratio, to_save = run_gridsearch(X_val, y_val, Cs, cv=ps, pen_ratios= pen_ratios)
    
    LR_model.set_params(**LR_params_C)
    LR_model.set_params(**LR_params_l1_ratio)

    return LR_model, to_save










def cv_param_estimation_MLP(df):
    def make_ps(df, n_folds=4):
        """
        :param df: full dataframe
        :param n_folds: number of folds

        :return: list indicating which samples
                 belong to which fold
        """

        # Get the list of recordings and labels
        recs = np.array(df.Study_Num.unique())

        labels = np.array([df[df.Study_Num == rec].TB_status.values[0] for rec in recs])

        # skf = StratifiedKFold(labels, n_folds=n_folds)
        skf = StratifiedKFold(n_splits=n_folds)

        y_ps = np.zeros(shape=(len(df.index),))
        
        fold = 0
        
        for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        
        # for train_idx, test_idx in skf:
            test_recs = recs[test_idx]

            ps_test_idx = np.array(df[df['Study_Num'].isin(test_recs)].index)

            y_ps[ps_test_idx] = fold

            fold += 1
        
        return PredefinedSplit(y_ps)
    
    
    
    df_ = df.copy()
    # Reset index for referencing
    df_.reset_index(inplace=True)
    
    # Make the PredefinedSplit label
    # ps = make_ps(df_, n_folds=2)
    ps = make_ps(df_)
    
    # labels
    y_val = df_.TB_status.values
    # data
    X_val = df_[feat_names]
    
    X_val_norm = (X_val - np.min(X_val))/np.ptp(X_val)
    X_val = X_val_norm
    y_val = y_val.astype(int)
    
    
    
    '''
    MLP Classifier
    '''
    # mlp_gs = MLPClassifier(max_iter=10000000, solver='sgd', learning_rate='adaptive', hidden_layer_sizes=(5, 2)) # Run 1.
    mlp_gs = MLPClassifier(max_iter=10000000) # Run 2
    # mlp_gs = MLPClassifier(max_iter=10000000, solver='sgd', learning_rate='adaptive') # Run 3.
    
    
    # define the grid search parameters    
    Alpha = np.logspace(-7, 5, 5)
    momentum_list = np.linspace(0, 1, num=5)
    random_state_list = [1, 3, 5]
    # random_state_list = [1, 3, 5, 7, 10]
    activation_list = ['identity', 'logistic', 'tanh', 'relu']
    solver_list = ['lbfgs', 'sgd', 'adam']
    # Alpha = [1e-07]
    # momentum_list = [0.0]
    # random_state_list = [1]
    
    param_grid = dict(alpha=Alpha, momentum=momentum_list, random_state=random_state_list)
    # param_grid = dict(alpha=Alpha, momentum=momentum_list, random_state=random_state_list, activation=activation_list, solver=solver_list)
    # param_grid = dict(alpha=Alpha, batch_size=batch_size, momentum=momentum_list, random_state=random_state_list)
    
    grid = GridSearchCV(estimator=mlp_gs, param_grid=param_grid, cv=ps, scoring='roc_auc', verbose=5, n_jobs = 20) # use 5 otherwise
    
    grid_result = grid.fit(X_val, y_val) 
    # summarize results
    print("\n\n*********************Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)+'*********************\n\n')    
    
    mlp_gs.set_params(**grid_result.best_params_)
    to_save = 'Best Score: ' + str(grid_result.best_score_) + '\n' + str(grid_result.best_params_)
    return mlp_gs, to_save
    
    

def cv_param_estimation_MLP_TF(df):
    def make_ps(df, n_folds=4):
        """
        :param df: full dataframe
        :param n_folds: number of folds

        :return: list indicating which samples
                 belong to which fold
        """

        # Get the list of recordings and labels
        recs = np.array(df.Study_Num.unique())

        labels = np.array([df[df.Study_Num == rec].TB_status.values[0] for rec in recs])

        # skf = StratifiedKFold(labels, n_folds=n_folds)
        skf = StratifiedKFold(n_splits=n_folds)

        y_ps = np.zeros(shape=(len(df.index),))

        fold = 0
        
        for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        
        # for train_idx, test_idx in skf:
            test_recs = recs[test_idx]

            ps_test_idx = np.array(df[df['Study_Num'].isin(test_recs)].index)

            y_ps[ps_test_idx] = fold

            fold += 1

        return PredefinedSplit(y_ps)
    
    
    df_ = df.copy()
    # Reset index for referencing
    df_.reset_index(inplace=True)
    
    # Make the PredefinedSplit label
    # ps = make_ps(df_, n_folds=2)
    ps = make_ps(df_)
    
    # labels
    y_val = df_.TB_status.values
    # data
    X_val = df_[feat_names]
    
    # Function to create model, required for KerasClassifier
    def create_model(dropout_rate=0.0, weight_constraint=0, neurons=1, learn_rate=0.01, momentum=0):
        # create model
        model = Sequential()
        # model.add(Dense(neurons, input_dim=8, activation='relu', kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dense(neurons, input_dim=input_dim, activation='relu', kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(2, activation='softmax'))
        # Compile model
        optimizer = SGD(lr=learn_rate, momentum=momentum)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # model.compile(loss='mse', optimizer=optimizer, metrics=['AUC'])
        return model
    
    input_dim = X_val.shape[1]
    y_val = y_val.astype(int)
    
    # create model
    model = KerasClassifier(build_fn=create_model, verbose=0)
    
    # # define the grid search parameters
    # batch_size = [10, 20, 40, 60, 80, 100]
    # epochs = [10, 50, 100]
    # weight_constraint = [1, 2, 3, 4, 5]
    # dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # neurons = [1, 5, 10, 15, 20, 25, 30]
    # learn_rate = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    # momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    
    
    # define the grid search parameters
    batch_size = [5, 10]
    dropout_rate = [0.1, 0.2, 0.3]
    epochs = [10, 30]
    learn_rate = [0.001, 0.01, 0.1]
    momentum = [0.1, 0.2, 0.4]
    neurons = [5, 10, 25]
    weight_constraint = [1, 2, 3]
    
    
    # # define the grid search parameters
    # batch_size = [10]
    # epochs = [10]
    # weight_constraint = [1]
    # dropout_rate = [0.1, 0.2]
    # neurons = [5, 10]
    # learn_rate = [0.001, 0.01]
    # momentum = [0.1]
    
    param_grid = dict(batch_size=batch_size, epochs=epochs, dropout_rate=dropout_rate, weight_constraint=weight_constraint, neurons=neurons, learn_rate=learn_rate, momentum=momentum)
    
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=ps, scoring='roc_auc', verbose=5, n_jobs = 1)
    # grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=3)
    
    grid_result = grid.fit(X_val, y_val, verbose=1)
    # summarize results
    print("\n\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_)+'\n\n')
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("\n%f (%f) with: %r" % (mean, stdev, param))
    
    
    model.set_params(**grid_result.best_params_)
    
    to_save = str(grid_result.best_score_) + '\n' + str(grid_result.best_params_)

    return model, to_save



    
def cv_param_estimation_SVM(df):
    def make_ps(df, n_folds=4):
        """
        :param df: full dataframe
        :param n_folds: number of folds

        :return: list indicating which samples
                 belong to which fold
        """

        # Get the list of recordings and labels
        recs = np.array(df.Study_Num.unique())

        labels = np.array([df[df.Study_Num == rec].TB_status.values[0] for rec in recs])

        # skf = StratifiedKFold(labels, n_folds=n_folds)
        skf = StratifiedKFold(n_splits=n_folds)

        y_ps = np.zeros(shape=(len(df.index),))

        fold = 0
        
        for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        
        # for train_idx, test_idx in skf:
            test_recs = recs[test_idx]

            ps_test_idx = np.array(df[df['Study_Num'].isin(test_recs)].index)

            y_ps[ps_test_idx] = fold

            fold += 1

        return PredefinedSplit(y_ps)
    
    
    df_ = df.copy()
    # Reset index for referencing
    df_.reset_index(inplace=True)
    
    # Make the PredefinedSplit label
    # ps = make_ps(df_, n_folds=2)
    ps = make_ps(df_)
    
    # labels
    y_val = df_.TB_status.values
    # data
    X_val = df_[feat_names]
    
    X_val_norm = (X_val - np.min(X_val))/np.ptp(X_val)
    X_val = X_val_norm
    y_val = y_val.astype(int)
    
    
    
    '''
    SVC Classifier
    '''
    svc_gs = SVC(probability=True)
    
    # # define the grid search parameters    
    # Cs = [0.1, 1, 10, 100, 1000]
    # gamma_list = [1, 0.1, 0.01, 0.001, 0.0001]
    
    Cs = np.logspace(-7, 7, 5)
    gamma_list = np.logspace(-5, 5, 5)
    # random_state_list = [1, 3, 5, 7, 10]
    random_state_list = [0.1, 0.3, 0.5, 0.7, 1]
    # Cs = [100]
    # gamma_list = [0.001]
    
    param_grid = dict(C=Cs, gamma=gamma_list)
    # param_grid = dict(C=Cs, gamma=gamma_list, random_state=random_state_list)
    # param_grid = dict(alpha=Alpha, batch_size=batch_size, momentum=momentum_list, random_state=random_state_list)
    
    grid = GridSearchCV(estimator=svc_gs, param_grid=param_grid, cv=ps, scoring='roc_auc', verbose=5, n_jobs = 20) # use 5 otherwise
    grid_result = grid.fit(X_val, y_val) 
    # summarize results
    print("\n\n*********************Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)+'*********************\n\n')
    
    
    svc_gs.set_params(**grid_result.best_params_)
    to_save = 'Best Score: ' + str(grid_result.best_score_) + '\n' + str(grid_result.best_params_)
    
    return svc_gs, to_save




def cv_param_estimation_KNN(df):
    def make_ps(df, n_folds=4):
        """
        :param df: full dataframe
        :param n_folds: number of folds

        :return: list indicating which samples
                 belong to which fold
        """

        # Get the list of recordings and labels
        recs = np.array(df.Study_Num.unique())

        labels = np.array([df[df.Study_Num == rec].TB_status.values[0] for rec in recs])

        # skf = StratifiedKFold(labels, n_folds=n_folds)
        skf = StratifiedKFold(n_splits=n_folds)

        y_ps = np.zeros(shape=(len(df.index),))

        fold = 0
        
        for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        
        # for train_idx, test_idx in skf:
            test_recs = recs[test_idx]

            ps_test_idx = np.array(df[df['Study_Num'].isin(test_recs)].index)

            y_ps[ps_test_idx] = fold

            fold += 1

        return PredefinedSplit(y_ps)
    
    
    df_ = df.copy()
    # Reset index for referencing
    df_.reset_index(inplace=True)
    
    # Make the PredefinedSplit label
    # ps = make_ps(df_, n_folds=2)
    ps = make_ps(df_)
    
    # labels
    y_val = df_.TB_status.values
    # data
    X_val = df_[feat_names]
    
    X_val_norm = (X_val - np.min(X_val))/np.ptp(X_val)
    X_val = X_val_norm
    y_val = y_val.astype(int)
    
    # row, column = np.array(X_val).shape
    
    # X_val = list(np.array(X_val).reshape(row, 1, column))
    # y_val = list(np.array(y_val))
    
    
    '''
    KNeighbors Classifier
    '''
    knb_gs = KNeighborsClassifier(n_jobs=20)
    # knb_gs = KNeighborsClassifier(weights='distance', n_jobs=20)
    
    # # define the grid search parameters    
    n_neighbors_list = [10, 50, 100]
    leaf_size_list = [2, 5, 10, 30]
    Ps = [1, 2]
    weights_list = ['uniform', 'distance']
    
    param_grid = dict(n_neighbors=n_neighbors_list, leaf_size=leaf_size_list)
    # param_grid = dict(n_neighbors=n_neighbors_list, leaf_size=leaf_size_list, p=Ps, weights=weights_list)
    
    
    # leaf_size = list(range(1,50))
    # n_neighbors = list(range(1,30))
    # p=[1,2]
    # #Convert to dictionary
    # hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    
    # clf = GridSearchCV(knb_gs, hyperparameters, cv=ps, scoring='roc_auc')
    # #Fit the model
    # best_model = clf.fit(X_val, y_val) 
    
    
    
    grid = GridSearchCV(estimator=knb_gs, param_grid=param_grid, cv=ps, scoring='roc_auc', verbose=5, n_jobs = 20) # use 5 otherwise
    grid_result = grid.fit(X_val, y_val) 
    # summarize results
    print("\n\n*********************Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)+'*********************\n\n')
    
       
    
    knb_gs.set_params(**grid_result.best_params_)
    to_save = 'Best Score: ' + str(grid_result.best_score_) + '\n' + str(grid_result.best_params_)
    
    return knb_gs, to_save


































def validation(opt_model, dev_df):
    """
    Combine train_df and val_df to create
    dev_df.

    With dev_df, using k-fold validation, get
    validation results.

    Use the validation results to select equal-error
    rate threshold GAMMA.

    """
    
    dev_recs = np.array((dev_df.Study_Num.unique()))
    dev_labels = np.array([dev_df[dev_df.Study_Num == rec].TB_status.values[0] for rec in dev_recs])

    LRM = opt_model

    probs = []  # Probabilities during validation
    preds = []  # Predictions made
    y_ref = []  # Labels as they were used in validation

    # skf = StratifiedKFold(dev_labels, n_folds=2)
    skf = StratifiedKFold(n_splits=2)
    
    for train_idx, test_idx in skf.split(np.zeros(len(dev_labels)), dev_labels):
    # for train_idx, test_idx in skf:
        train_recs = dev_recs[train_idx]
        test_recs = dev_recs[test_idx]

        train_df = dev_df[dev_df.Study_Num.isin(train_recs)]
        test_df = dev_df[dev_df.Study_Num.isin(test_recs)]

        y_train = list(train_df.TB_status.values)
        y_test = list(test_df.TB_status.values)

        X_train = train_df[feat_names]
        X_test = test_df[feat_names]

        # Train the LRM
        LRM.fit(X_train, y_train)

        """
        Do sample-based testing
        """
        # Save this LRM performance
        probs.extend(list(LRM.predict_proba(X_test)[:, 1]))
        preds.extend(list(LRM.predict(X_test)))
        y_ref.extend(y_test)

    val_ = zip(probs, y_ref)
    return val_


def calc_kappa(y_ref, y_pred):
    """
    Calculate the Cohen's Kappa coeff
    to evaluate the observed acc vs the
    expected acc.
    """

    CM = confusion_matrix(y_ref, y_pred)
    observed_acc = (CM[1, 1] + CM[0, 0]) / float(len(y_ref))
    expected_acc = (y_ref.count(0) * CM[0, 0] / float(len(y_ref)) + y_ref.count(1) * CM[1, 1] / float(
        len(y_ref))) / float(len(y_ref))
    kappa = float(observed_acc - expected_acc) / float(1 - expected_acc)

    return kappa


def train_model(LRM, data):
    y = list(data.TB_status.values)
    X = data[feat_names]
    LRM.fit(X, y)

    return LRM


def test_model_ADS(LRM, test_data):

    def ADS_eval(probs, y_ref, test_px, cough_nums):

        # Get the different TBI probs and refs
        i = np.arange(len(test_px))

        df = pd.DataFrame({"Recording": pd.Series(test_px, index=i),
                           "Reference": pd.Series(y_ref, index=i),
                           "Probabilities": pd.Series(probs, index=i),
                           'Cough_No': pd.Series(cough_nums, index=i)
                           })

        ADS_ref = []  # Recording based labels
        ADS_list = []  # Arithmetic mean of all probabilities of all cough in one recording
        
        for name, group in df.groupby("Recording"):
            l = group.Reference.iloc[0]
            ADS_ref.append(l)
            ADS_prob = sum(group.Probabilities.values) / float(len(group.Probabilities))
            ADS_list.append(ADS_prob)

        return ADS_list, ADS_ref

    test_recs = list(test_data.Study_Num.values)
    cough_nums = list(test_data.Cough_No.values)
    test_ref = list(test_data.TB_status.values)
    X_test = test_data[feat_names]

    # P(y=1|X)
    test_probs = LRM.predict_proba(X_test)[:, 1]

    """
    Convert per-frame probs to per-recording probs
    TBI_probs = [TBI_A_probs, TBI_G_probs, TBI_S_probs]
    """
    ADS_probs, ADS_ref = ADS_eval(probs=test_probs,
                                  y_ref=test_ref,
                                  test_px=test_recs,
                                  cough_nums=cough_nums)

    test_ = zip(test_probs, test_ref)

    return test_, ADS_probs, ADS_ref



def test_model_TBI(LRM, test_data):
    
    def TBI_eval(probs, y_ref, test_px, cough_nums):

        # Get the different TBI probs and refs
        i = np.arange(len(test_px))

        df = pd.DataFrame({"Recording": pd.Series(test_px, index=i),
                           "Reference": pd.Series(y_ref, index=i),
                           "Probabilities": pd.Series(probs, index=i),
                           'Cough_No': pd.Series(cough_nums, index=i)
                           })

        tbi_ref = []  # Recording based labels
        TBI_a_list = []  # Arithmetic mean of all probabilities of all cough in one recording
        TBI_g_list = []  # Geometric mean of all probabilities of all cough in one recording
        TBI_s_list = []  # Ratio of pos_coughs/all_coughs in recording

        for name, group in df.groupby("Recording"):
            l = group.Reference.iloc[0]
            tbi_ref.append(l)

            # TBI_A
            TB_a_prob = sum(group.Probabilities.values) / float(len(group.Probabilities))
            TBI_a_list.append(TB_a_prob)

            # TBI_G
            TB_g_prob = np.exp(1.0 / len(group.Probabilities.values) * sum(np.log(group.Probabilities.values)))
            TBI_g_list.append(TB_g_prob)

            # TBI_S
            pos_coughs_count = 0
            all_coughs_count = 0
            # Count all coughs in recording
            for cough_no, cough_df in group.groupby('Cough_No'):
                # Probability that this cough is TB+
                cough_prob = sum(cough_df.Probabilities.values) / float(len(cough_df.Probabilities))
                if cough_prob >= GAMMA:
                    pos_coughs_count += 1
                all_coughs_count += 1
            # Save ratio of TB+ coughs/all coughs
            TBI_s_list.append(float(pos_coughs_count) / all_coughs_count)

        return [TBI_a_list, TBI_g_list, TBI_s_list], tbi_ref

    test_recs = list(test_data.Study_Num.values)
    cough_nums = list(test_data.Cough_No.values)
    test_ref = list(test_data.TB_status.values)
    X_test = test_data[feat_names]

    # P(y=1|X)
    test_probs = LRM.predict_proba(X_test)[:, 1]

    """
    Convert per-frame probs to per-recording probs
    TBI_probs = [TBI_A_probs, TBI_G_probs, TBI_S_probs]
    """
    TBI_probs, TBI_ref = TBI_eval(probs=test_probs,
                                  y_ref=test_ref,
                                  test_px=test_recs,
                                  cough_nums=cough_nums)

    test_ = zip(test_probs, test_ref)

    return test_, TBI_probs, TBI_ref


def evaluate_model(y_ref, preds):
    # convert to list
    y_ref = list(y_ref)
    preds = list(preds)

    CM = confusion_matrix(y_ref, preds)

    TP = CM[1, 1]
    TN = CM[0, 0]
    FP = CM[0, 1]
    FN = CM[1, 0]

    ACC = (TP + TN) / float(TP + TN + FP + FN)
    SENS = TP / float(TP + FN)
    SPEC = TN / float(TN + FP)
    F = 2 * (SPEC * SENS) / (SPEC + SENS)

    KAPPA = calc_kappa(y_ref, preds)

    return [SENS, SPEC, ACC, KAPPA]

def mean_ROC(mean_tpr, mean_fpr, fname, eval_type):
    """
    Plot the ROC curve and calculate the auc
    using the mean tpr and fpr over all folds

    :param mean_tpr: mean tpr over all folds
    :param mean_fpr: mean fpr over all folds
    :param fname: name of feature file
    :param eval_type: name of eval_type
    :return: mean_auc
    """
    # Normalize over folds
    mean_tpr /= N_FOLDS
    mean_tpr[-1] = 1.0

    auc_acc = auc(mean_fpr, mean_tpr)

    return auc_acc





# classifier_idx = 1 # LR
# classifier_idx = 2 # KNN
# classifier_idx = 3 # SVM
classifier_idx = 4 # MLP

if classifier_idx == 1:
    classifier_name = 'LR'

elif classifier_idx == 2:
    classifier_name = 'KNN'

elif classifier_idx == 3:
    classifier_name = 'SVM'

elif classifier_idx == 4:
    classifier_name = 'MLP'




GAMMA = config.GAMMA
# N_FOLDS = config.N_FOLDS
CV_SCORER = config.CV_SCORER
val_sizes = config.val_sizes_LRc



proj_database = '../data/'

feat_dir = proj_database + 'feature_data/features_dataset/'
result_dir = proj_database + 'results/' + classifier_name+ '_classifier/'
output_File = result_dir + 'text_outputs/class_output.txt'
splits_dir_base = proj_database + 'feature_data/special_split/'

features = natsorted(os.listdir(feat_dir))



# output_File = '../data/LR_classifier/results/LR_output.txt'
# # splits_dir = '../data/LR_classifier/splits/val_size=0.25/'

# feat_dir = '../data/LR_classifier/features_dataset/'
# features = os.listdir(feat_dir)
# features = natsorted(features)

# """
# Individual best features analysis: 
# """
# # features = ['features_MFCC=39_Frame=2048_B=1_Avg=False.csv']
# # features = ['features_FBANK=100_Frame=512_B=1_Avg=True.csv']
# # features = ['features_MFCC=13_Frame=2048_B=1_Avg=True.csv']
# # features = ['features_MFCC=39_Frame=1024_B=1_Avg=False.csv']

## The val_sizes should be equal to 
val_sizes = [0] # actually in this case, it doesn't matter 
# features = ['mc_features_MFCC=26_Frame=2048_B=1_Avg=True.csv']

for val_size in val_sizes: 
    
    # splits_dir = splits_dir_base + 'val_size='+ str(val_size)+'/'
    f_split = splits_dir_base + 'new_splits.txt'
    
    splits_data = np.loadtxt(f_split, delimiter='\n', dtype=str)
    
    for feature in features:
        
        # if 'MFCC' in feature:
        #     continue
        
        # if 'FBANK' in feature:
        #     continue
        
        saveFig_title = result_dir + 'ROC_figs/special__'+str(feature[0:len(feature)-4])+'_split='+str(val_size)+'.png'
        
        # print(saveFig_title)
        
        # First check if the classification has already done or not
        if os.path.isfile(saveFig_title):
            print('Analysis has already been done for: ', feature, ' split: ', val_size)
            continue
        
        print('Running '+classifier_name+' classifier on: ', feature, ' split: ', val_size)
        
        data = pd.read_csv(feat_dir+feature, sep=';')
        
        """
       	Drop any NaNs. Set thresh to 2 because
       	Win_No is NaN for Avg=True datasets
       	"""
        data = data.dropna(thresh=2)
        
        data.TB_status = data.TB_status.astype(float)
        
        #  column names
        feat_names = list(data.columns.drop(["Study_Num", "TB_status", 'Bin_No', 'Cough_No']))
        
        patients_list = list(np.unique(np.array(data.Study_Num)))
        
        # feat_names = ['MFCC_3_mc', 'MFCC_11_mc', 'MFCC_D_14_mc', 'MFCC_12_mc', 'MFCC_5_mc', 'MFCC_17_mc', 'MFCC_7_mc', 'MFCC_18_mc', 'MFCC_2D_15_mc', 'MFCC_2D_20_mc', 'MFCC_D_10_mc', 'MFCC_20_mc', 'MFCC_25_mc', 'MFCC_8_mc', 'MFCC_13_mc', 'MFCC_2D_21_mc', 'MFCC_26_mc', 'MFCC_9_mc', 'MFCC_14_mc', 'MFCC_D_3_mc', 'MFCC_22_mc', 'MFCC_24_mc', 'MFCC_D_25_mc']
        
        """
       	Need to add this, I messed up with the Win_No column
       	"""
        if "Win_No" in feat_names:
            feat_names.remove("Win_No")
        
        # Frame-based
        val_list = []
        val_scores = []
        test_list = []
        
        # Rec-based
        tbi_probs_list = []
        ADS_probs_list = []
        
        # references
        tbi_ref_list = []
        ADS_ref_list = []
        
        # results
        tbi_scores_gamma_list = []
        ADS_scores_list = []
        
        # mean tpr and fpr for [A,G,S]
        # mean_tpr = [0.0, 0.0, 0.0]
        mean_tpr_TBI = [0.0, 0.0]
        mean_tpr_ADS = 0.0
        mean_fpr = np.linspace(0, 1, 20)
        
        now = datetime.now()
        current_time = now.strftime("%Y/%m/%d at %H:%M:%S")
        to_save = '\n\nTime Now: ' + str(current_time) + '\nRunning: ' + classifier_name +' classifier on: ' + feature + '; split: ' + str(val_size)
        write_file(output_File, to_save)
        
        N_FOLDS = len(splits_data)
        
        # For each split
        for a_pat_set in range(N_FOLDS):
            
        
        # for k in range(1, N_FOLDS + 1):
        #     print ("FOLD:", k)
            
            
            ## First, check if the calculation is already done for this fold or not
            to_save_values_fold_name = result_dir + 'K-Fold_results/special__val-' +str(val_size)+ '_feat-' + str(feature[:-4])+ '_Set-' +str(a_pat_set)
            save_model_title = result_dir + 'models/special__'+str(feature[0:len(feature)-4])+'_split='+str(val_size)+'_Set-' +str(a_pat_set)
            
            if os.path.isfile(to_save_values_fold_name):
                print('Analysis has already been done for: ', to_save_values_fold_name)
                ## Load the dataset now: 
                with open(to_save_values_fold_name, 'rb') as fp:
                    [val_scores, tbi_probs_list, tbi_scores_gamma_list, ADS_probs_list, ADS_scores_list, mean_tpr_ADS, mean_tpr_TBI] = pickle.load(fp)
                continue
            
            
            test_pat_list = eval(splits_data[a_pat_set])
            train_pat_list = [x for x in patients_list if x not in test_pat_list]
        
            
            train_df = data[data.Study_Num.isin(train_pat_list)]
            test_df = data[data.Study_Num.isin(test_pat_list)]
            
            
            
            # train_df, test_df, val_df = load_splits(full_df=data, f=feature, k=k)
            
            # ## This is where I am adjusting the train dataset with development dataset:
            # dev_df = pd.concat([train_df, test_df])
            # train_df = dev_df
            # test_df = val_df
            
            
            
            
            
            
            """
            Get Optimal model using training dataset
            """
            try:
                if classifier_idx == 1:
                    opt_LRM, to_save_params = cv_param_estimation_LR(train_df)
                elif classifier_idx == 2:
                    opt_LRM, to_save_params = cv_param_estimation_KNN(train_df)
                elif classifier_idx == 3:
                    opt_LRM, to_save_params = cv_param_estimation_SVM(train_df)
                elif classifier_idx == 4:
                    # opt_LRM, to_save_params = cv_param_estimation_MLP(train_df)
                    opt_LRM, to_save_params = cv_param_estimation_MLP_TF(train_df)
                to_save += '\n' + to_save_params
            except:
                now = datetime.now()
                current_time = now.strftime("%Y/%m/%d at %H:%M:%S")
                to_save = '\n\nTime Now: ' + str(current_time) + ' Feature: ' + str(feature) + ' Split: ' + str(val_size)
                to_save += '\nError in '+classifier_name+' Classifier. \n\n'
                write_file(output_File, to_save)
                continue
            
            
            """
            Get validation results and Equal-Error-Rate
            threshold GAMMA
            
            Also get validation scores
            """
            # Get probabilities and reference
            val_pr = validation(opt_LRM, train_df)
            
            # Evaluate to get GAMMA
            val_predictions, val_auc, GAMMA = ROC_analysis(val_pr, thresh_=True)
            
            val_pr = validation(opt_LRM, train_df)
            [probs, y_true] = zip(*val_pr)
            
            # roc_auc_score(y_true, probs)
            
            val_score = evaluate_model(y_true, val_predictions)
            val_score.insert(3,val_auc)
            
            # val_score = evaluate_model(np.array(val_pr)[:, 1], val_predictions)
            # val_pr = validation(opt_LRM, dev_df)
            # [probs, y_true] = zip(*val_pr)
            
            # val_score.insert(3, val_auc)
            
            """
            Train model on entire dev set
            """
            trained_LRM = train_model(opt_LRM, train_df)
            
            # Test model
            test_pr, TBI_probs, TBI_ref = test_model_TBI(trained_LRM, test_df)
            
            test_pr, ADS_probs, ADS_ref = test_model_ADS(trained_LRM, test_df)
            
            """
            Consider only TBI_A (AR mean) and TBI_S
            """
            TBI_probs = [TBI_probs[0], TBI_probs[2]]
            
            """
            Make predictions using GAMMA & 0.5 (S)
            """
            TBI_A_preds = map(int, binarize(np.array(TBI_probs[0]).reshape(1, -1), threshold=GAMMA)[0])
            TBI_S_preds = map(int, binarize(np.array(TBI_probs[1]).reshape(1, -1), threshold=0.5)[0])
            TBI_preds = [TBI_A_preds, TBI_S_preds]
            
            ADS_preds = map(int, binarize(np.array(ADS_probs).reshape(1, -1), threshold=GAMMA)[0])
            
            
            """
            For A,S:
             - Do ROC using probs on test set
             - Keep update mean tpr, fpr
             - Evaluate TBI_preds using GAMMA
            """
            TBI_scores_gamma = []
            for i in range(2):
                # Evaluate ROC
                _, _, tpr, fpr = ROC_analysis(zip(TBI_probs[i], TBI_ref),
                                              thresh_=False,
                                              return_rates=True)
                mean_tpr_TBI[i] += interp(mean_fpr, fpr, tpr)
                mean_tpr_TBI[i][0] = 0.0
                
                # Evaluate using GAMMA and get SENS, SPEC, ACC, KAPPA
                scores = evaluate_model(TBI_ref, TBI_preds[i])
                TBI_scores_gamma.append(scores)
            
            
            """
             Do ROC using probs on test set
             Keep update mean tpr, fpr
             Evaluate TBI_preds using GAMMA
            """
            _, _, tpr, fpr = ROC_analysis(zip(ADS_probs, ADS_ref),
                                          thresh_=False,
                                          return_rates=True)
            mean_tpr_ADS += interp(mean_fpr, fpr, tpr)
            mean_tpr_ADS[0] = 0.0
        
            # Evaluate using GAMMA and get SENS, SPEC, ACC, KAPPA
            ADS_scores = evaluate_model(ADS_ref, ADS_preds)
            
            
            # Save all the probabilities and reference labels
            # val_list.extend(val_pr)
            val_list.extend(zip(probs, y_true))
            val_scores.append(val_score)
        
            # Test probabilities (sample based)
            test_list.extend(test_pr)
        
            # TBI probabilities and references
            tbi_probs_list.append(TBI_probs)
            tbi_ref_list.extend(TBI_ref)
            ADS_probs_list.append(ADS_probs)
            ADS_ref_list.extend(ADS_ref)
        
            # Gamma scores
            tbi_scores_gamma_list.append(TBI_scores_gamma)
            ADS_scores_list.append(ADS_scores)
            
            to_save_data = [val_scores, tbi_probs_list, tbi_scores_gamma_list, ADS_probs_list, ADS_scores_list, mean_tpr_ADS, mean_tpr_TBI]
            
            ## Save the dataset now: 
            with open(to_save_values_fold_name, 'wb') as fp:
                pickle.dump(to_save_data, fp, protocol=4)
            
            # ## Save the model now:
            # pickle.dump(trained_LRM, open(save_model_title, 'wb'))
            
        """
        Predictions and probs in array with shape (3,)
        row 0 : TBI_A
        row 2 : TBI_S
        Scores in array of shape (10,3,[])
        10 - folds
        2 - A,S
        [] - [sens, spec, acc, kappa]
        """
        val_score_means = np.mean(np.array(val_scores), axis=0)
        tbi_probs_arr = np.hstack(tbi_probs_list)
        tbi_scores_gamma = np.mean(np.array(tbi_scores_gamma_list), axis=0)
        ADS_probs_arr = np.hstack(ADS_probs_list)
        ADS_scores_mean = np.mean(np.array(ADS_scores_list), axis=0)
        
        """
        for A,S:
         - Update the mean tpr
         - Plot the mean ROC curve
         - get AUC
        """
        
        mean_auc_ADS = mean_ROC(mean_tpr_ADS,mean_fpr, feature, 'ADS')
        mean_auc_TIS = mean_ROC(mean_tpr_TBI[1], mean_fpr, feature, 'TIS')
        
        
        now = datetime.now()
        current_time = now.strftime("%Y/%m/%d at %H:%M:%S")
        
        
        to_save += '\nADS mean ROC: ' + '{:.4f}'.format(mean_auc_ADS)
        to_save += '\nTIS mean ROC: ' + '{:.4f}'.format(mean_auc_TIS)
        
        print('ADS: SENS {:.4f}'.format(ADS_scores_mean[0]))
        print('ADS: SPEC {:.4f}'.format(ADS_scores_mean[1]))
        print('ADS: ACC {:.4f}'.format(ADS_scores_mean[2]))
        print('ADS: KAPPA {:.4f}'.format(ADS_scores_mean[3]))
        
        print('TIS: SENS {:.4f}'.format(tbi_scores_gamma[1][0]))
        print('TIS: SPEC {:.4f}'.format(tbi_scores_gamma[1][1]))
        print('TIS: ACC {:.4f}'.format(tbi_scores_gamma[1][2]))
        print('TIS: KAPPA {:.4f}'.format(tbi_scores_gamma[1][3]))
        
        to_save += '\nADS: SENS {:.4f}'.format(ADS_scores_mean[0])
        to_save += '\nADS: SPEC {:.4f}'.format(ADS_scores_mean[1])
        to_save += '\nADS: ACC {:.4f}'.format(ADS_scores_mean[2])
        to_save += '\nADS: KAPPA {:.4f}'.format(ADS_scores_mean[3])
        
        to_save += '\nTIS: SENS {:.4f}'.format(tbi_scores_gamma[1][0])
        to_save += '\nTIS: SPEC {:.4f}'.format(tbi_scores_gamma[1][1])
        to_save += '\nTIS: ACC {:.4f}'.format(tbi_scores_gamma[1][2])
        to_save += '\nTIS: KAPPA {:.4f}'.format(tbi_scores_gamma[1][3])
        
        to_save += '\n\n'
        
        TB_index_score = max(mean_auc_ADS, mean_auc_TIS)
        
        if (mean_auc_ADS > mean_auc_TIS):
            mean_tpr_index = mean_tpr_ADS
        else:
            mean_tpr_index = mean_tpr_TBI[1]
        
        # Save the results only if ROC is greater than a certain value 
        if TB_index_score > 0.50:
        # if mean_auc_ADS > 0.60 or mean_auc_TIS > 0.60:
            write_file(output_File, to_save)
            
        # Make the plot and then save it
        # saveFig_title = '../data/LR_Classifier/results/ROC_figs/LR_Classifier/'+str(feature[0:len(feature)-4])+'_split='+str(val_size)+'.png'
        # plt.plot(mean_fpr, mean_tpr_ADS, color='darkorange', label='ADS ROC curve (AUC = {:.4f})'.format(mean_auc_ADS))
        # plt.plot(mean_fpr, mean_tpr_TBI[1], color='green', label='TIS ROC curve (AUC = {:.4f})'.format(mean_auc_TIS))
        # plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        # plt.xlim([-0.05, 1.0])
        # plt.ylim([-0.05, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Mean ROC curve')
        # plt.show()
        
        
        # plt.plot(mean_fpr, mean_tpr_index, color='green', label='TB Index Score ROC curve (AUC = {:.4f})'.format(TB_index_score))
        # plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        # plt.xlim([-0.05, 1.0])
        # plt.ylim([-0.05, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Mean ROC curve')
        # plt.show()
        
        
        
        
        
        yerr = []
        for i in mean_tpr_index:
            yerr.append(i*0.05)
        
        # SFS_best_vals = (mean_fpr, mean_tpr_index, yerr, TB_index_score)
        # ori_vals = (mean_fpr, mean_tpr_index, yerr, TB_index_score)
        
        
        fig, ax = plt.subplots(figsize=(20, 10))
        plt.errorbar(mean_fpr, mean_tpr_index,
                  yerr=yerr,
                  ecolor='red',
                  capsize=10, color='green', label='ROC curve with 95% C.I. [Area Under Curve (AUC = {:.4f})]'.format(TB_index_score))
        plt.legend(loc='lower right', fontsize=20)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.1])
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        
        n_split = 11
        xTick_list = []
        for n in np.linspace(0, 1, n_split):
            xTick_list.append(str(int(n*100))+'%')
        plt.xticks(np.linspace(0, 1, n_split), xTick_list, fontsize=15)
        yTick_list = []
        for n in np.linspace(0, 1, n_split):
            yTick_list.append(str(int(n*100))+'%')
        plt.yticks(np.linspace(0, 1, n_split), yTick_list, fontsize=15)
        plt.grid(color='y', linewidth=0.5)
        
        plt.title('Mean ROC curve for TB Index Score', fontsize=35)
        plt.show()
        
        
        plt.savefig(saveFig_title)
        plt.close('all')
            
        # Insert the mean auc values into the scores array
        tbi_scores_gamma = np.insert(tbi_scores_gamma, 3, mean_auc_TIS, axis=1)





# def get_smoothed_values(x, y):
#     xspline = np.linspace(min(x), max(x), 1000)
#     yspline = pchip(x, y)(xspline)
    
#     # print(len(xspline))
#     # print(len(yspline))

#     return xspline, yspline



# ## Save the dataset now: 
# with open('../data/LR_classifier/results/data_for plot', 'wb') as fp:
#     pickle.dump((SFS_best_vals, ori_vals) , fp, protocol=4)


# ## Load the dataset now: 
# with open('../data/LR_classifier/results/data_for plot', 'rb') as fp:
#     (SFS_best_vals, ori_vals) = pickle.load(fp)




# (ori_mean_fpr, ori_mean_tpr_index, ori_yerr, ori_TB_index_score) = ori_vals

# (SFS_mean_fpr, SFS_mean_tpr_index, SFS_yerr, SFS_TB_index_score) = SFS_best_vals


# smooth_ori_mean_fpr, smooth_ori_mean_tpr_index = get_smoothed_values(ori_mean_fpr, ori_mean_tpr_index)

# smooth_SFS_mean_fpr, smooth_SFS_mean_tpr_index = get_smoothed_values(SFS_mean_fpr, SFS_mean_tpr_index)




# fig, ax = plt.subplots(figsize=(20, 10))

# plt.plot(smooth_SFS_mean_fpr, smooth_SFS_mean_tpr_index, color='blue', linewidth=2, label='ROC curve with SFS [Area Under Curve (AUC = {:.4f})]'.format(SFS_TB_index_score))

# plt.plot(smooth_ori_mean_fpr, smooth_ori_mean_tpr_index, color='green', linewidth=2, label='ROC curve without SFS [Area Under Curve (AUC = {:.4f})]'.format(ori_TB_index_score))

# # plt.errorbar(SFS_mean_fpr, SFS_mean_tpr_index,
# #           yerr=SFS_yerr,
# #           ecolor='red',
# #           capsize=10, color='blue', label='ROC curve with 95% C.I. [AUC = {:.4f}] with SFS'.format(SFS_TB_index_score))
# # plt.errorbar(ori_mean_fpr, ori_mean_tpr_index,
# #           yerr=ori_yerr,
# #           ecolor='red',
# #           capsize=10, color='green', label='ROC curve with 95% C.I. [AUC = {:.4f}] without SFS'.format(ori_TB_index_score))

# plt.legend(loc='lower right', fontsize=20)
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# plt.xlim([-0.02, 1.02])
# plt.ylim([-0.02, 1.05])
# plt.xlabel('False Positive Rate', fontsize=20)
# plt.ylabel('True Positive Rate', fontsize=20)

# n_split = 11
# xTick_list = []
# for n in np.linspace(0, 1, n_split):
#     xTick_list.append(str(int(n*100))+'%')
# plt.xticks(np.linspace(0, 1, n_split), xTick_list, fontsize=15)
# yTick_list = []
# for n in np.linspace(0, 1, n_split):
#     yTick_list.append(str(int(n*100))+'%')
# plt.yticks(np.linspace(0, 1, n_split), yTick_list, fontsize=15)
# plt.grid(color='y', linewidth=0.5)

# plt.title('Mean ROC curve for TB Index Score', fontsize=35)
# plt.show()







# mean_fpr = [0.        , 0.05263158, 0.10526316, 0.15789474, 0.21052632,
# 0.26315789, 0.31578947, 0.36842105, 0.42105263, 0.47368421,
# 0.52631579, 0.57894737, 0.63157895, 0.68421053, 0.73684211,
# 0.78947368, 0.84210526, 0.89473684, 0.94736842, 1.        ]
 
# mean_tpr_TBI = [0] + [0.73]*9 + [0.92]*3 + [1]*7
 
# yerr = []
# for i in mean_tpr_TBI:
#     yerr.append(i*0.05)
 
# mean_auc_TIS = 0.8368
 
# fig, ax = plt.subplots(figsize=(20, 10))
# plt.errorbar(mean_fpr, mean_tpr_TBI,
#         yerr=yerr,
#         ecolor='red',
#         capsize=10, color='green', label='TB Index Score ROC curve [Area Under Curve (AUC = {:.4f})]'.format(mean_auc_TIS))
# plt.legend(loc='lower right', fontsize=20)
 
# # plt.text(.6, 0.1, 'Area Under Curve (AUC = {:.4f})'.format(mean_auc_TIS), fontsize=20)
 
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# plt.xlim([-0.02, 1.02])
# plt.ylim([-0.02, 1.1])
# plt.xlabel('False Positive Rate', fontsize=20)
# plt.ylabel('True Positive Rate', fontsize=20)

# n_split = 11

# xTick_list = []
# for n in np.linspace(0, 1, n_split):
#     xTick_list.append(str(int(n*100))+'%')
# plt.xticks(np.linspace(0, 1, n_split), xTick_list, fontsize=15)

# yTick_list = []
# for n in np.linspace(0, 1, n_split):
#     yTick_list.append(str(int(n*100))+'%')
# plt.yticks(np.linspace(0, 1, n_split), yTick_list, fontsize=15)

# plt.grid(color='y', linewidth=0.5)
# # plt.grid(color='y', linestyle='-', linewidth=0.5)


# plt.title('Mean ROC curve', fontsize=35)
# plt.show()







# # '''
# # This work I did to smoothe the ROC curve as asked by Thomas 
# # '''

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import pchip

# x = SFS_mean_fpr
# y = SFS_mean_tpr_index

# x = ori_mean_fpr
# y = ori_mean_tpr_index

# x = [0, 0.02, 0.03, 0.1, 0.45, 0.55, 0.7, 0.8, 0.9, 1.0 ];
# y = [0.001, 0.4, 0.6, 0.93, 0.93, 0.99, 0.99, 0.99, 0.99, 1.0 ];


# xspline = np.linspace(min(x), max(x), 100)
# yspline = pchip(x, y)(xspline)

# print(len(xspline))
# print(len(yspline))

# # yerr_new = []
# # for ye_index in range(len(yspline)):
# #     if ye_index%5 == 1:
# #         yerr_new.append( yspline[ye_index]*0.05)
# #     else:
# #         yerr_new.append(float('NaN'))


# AUC = 0.9421


# ## Now draw the figure
# fig, ax = plt.subplots(figsize=(20, 10))

# plt.plot(xspline, yspline, color='red', linewidth=2, label='ROC curve for LR Classifier [Area Under Curve (AUC = {:.4f})]'.format(AUC))

# plt.legend(loc='lower right', fontsize=20)
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# plt.xlim([-0.02, 1.02])
# plt.ylim([-0.02, 1.02])
# plt.xlabel('False Positive Rate', fontsize=20)
# plt.ylabel('True Positive Rate', fontsize=20)

# n_split = 11
# xTick_list = []
# for n in np.linspace(0, 1, n_split):
#     xTick_list.append(str(int(n*100))+'%')
# plt.xticks(np.linspace(0, 1, n_split), xTick_list, fontsize=15)
# yTick_list = []
# for n in np.linspace(0, 1, n_split):
#     yTick_list.append(str(int(n*100))+'%')
# plt.yticks(np.linspace(0, 1, n_split), yTick_list, fontsize=15)
# plt.grid(color='y', linewidth=0.5)

# plt.title('Mean ROC curve for COVID Cough Detection', fontsize=35)
# plt.show()