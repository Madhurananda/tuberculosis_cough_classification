#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:00:11 2020

@author: madhu
"""

import sys
import numpy as np
import pandas as pd
# import helper_stratified as help
from operator import itemgetter
from numpy import interp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import config
import os
from sklearn.preprocessing import binarize, scale, robust_scale
import pickle
from natsort import natsorted

# from LR_Classifier_optimiser import load_splits, write_file, ROC_analysis, validation, calc_kappa, train_model, test_model_ADS, test_model_TBI, evaluate_model, mean_ROC 
# sys.path.insert(0, './Renier_scripts')
# from helper import *
from sklearn.metrics import *

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, PredefinedSplit



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


def load_splits(full_df, f, k):
    """
	full_df:	Complete dataframe
	f:			Name of feature dataset (csv)
	k:			Split number
	"""
    f_split = "".join([splits_dir, str(k), '/', os.path.basename(f)[:-4], ".txt"])

    splits_data = np.loadtxt(f_split, delimiter='=', dtype=str)

    train_recs = eval(splits_data[0, 1])
    test_recs = eval(splits_data[1, 1])
    val_recs = eval(splits_data[2, 1])
    
    train_df = full_df[full_df.Study_Num.isin(train_recs)]
    test_df = full_df[full_df.Study_Num.isin(test_recs)]
    val_df = full_df[full_df.Study_Num.isin(val_recs)]
    
    return train_df, test_df, val_df




def validation(opt_model, dev_df, feat_names):
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


def train_model(LRM, data, feat_names):
    y = list(data.TB_status.values)
    X = data[feat_names]
    LRM.fit(X, y)

    return LRM


def test_model_ADS(LRM, test_data, feat_names):

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



def test_model_TBI(LRM, test_data, feat_names):
    
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








def find_AUC(data, feat_names):
    
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
    
    # now = datetime.now()
    # current_time = now.strftime("%Y/%m/%d at %H:%M:%S")
    # to_save = '\n\nTime Now: ' + str(current_time) + ' Feature: ' + str(feature) + ' Split: ' + str(val_size)
    
    # For each split
    for k in range(1, N_FOLDS + 1):
        # print ("FOLD:", k)
        train_df, test_df, val_df = load_splits(full_df=data, f=feature, k=k)
        
        ## This is where I am adjusting the train dataset with development dataset:
        dev_df = pd.concat([train_df, test_df])
        train_df = dev_df
        test_df = val_df
        
        # feat_names = feat_names + ['Study_Num']
        # feat_names = feat_names + ['Study_Num', 'TB_status']
                
        # train_df = dev_df[feat_names]
        # test_df = val_df[feat_names]        
        
        """
        Get Optimal LR model using training dataset
        """
        opt_LRM, to_save_params = cv_param_estimation(train_df, feat_names)
        
        """
        Get validation results and Equal-Error-Rate
        threshold GAMMA
        
        Also get validation scores
        """
        # Get probabilities and reference
        val_pr = validation(opt_LRM, train_df, feat_names)
        
        # Evaluate to get GAMMA
        val_predictions, val_auc, GAMMA = ROC_analysis(val_pr, thresh_=True)
        
        val_pr = validation(opt_LRM, train_df, feat_names)
        [probs, y_true] = zip(*val_pr)
        val_score = evaluate_model(y_true, val_predictions)
        val_score.insert(3,val_auc)
        
        
        """
        Train model on entire dev set
        """
        trained_LRM = train_model(opt_LRM, train_df, feat_names)
        
        # Test model
        test_pr, TBI_probs, TBI_ref = test_model_TBI(trained_LRM, test_df, feat_names)
        
        test_pr, ADS_probs, ADS_ref = test_model_ADS(trained_LRM, test_df, feat_names)
        
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
    
    ## This needs to be changed for TIS or ADS 
    return mean_auc_ADS
    




def cv_param_estimation(df, feat_names):
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


        grid_search = LogisticRegressionCV(Cs=Cs, cv=cv, scoring='roc_auc', max_iter = 100000000, solver='saga', l1_ratios = pen_ratios, penalty='elasticnet', n_jobs = 20)
        
        # Scale the data
        # X_scaled = scale(X)
        
        X_r_scaled = robust_scale(X)
        
        # Fit GS with this data
        grid_search.fit(X_r_scaled, y)
        
        gs_opt_params_C = {'C': grid_search.C_[0]}
        
        gs_opt_params_l1_ratio = {'l1_ratio': grid_search.l1_ratio_[0]}
        
        # gs_opt_params = grid_search.best_params_
        
        # print (gs_opt_params_C)
        # print(gs_opt_params_l1_ratio)
        
        '''
        LogisticRegressionCV.scores_ gives the score for all the folds.
        GridSearchCV.best_score_ gives the best mean score over all the folds.
        '''
        
        # print ('Local Max auc_roc:', grid_search.scores_[1].max())  # is wrong
        # print ('Max auc_roc:', grid_search.scores_[1].mean(axis=0).max())  # is correct
        
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
    # LR_model = LogisticRegression(max_iter=100000000, solver='saga', penalty='elasticnet', class_weight='balanced', n_jobs = -1)
    
    Cs = np.logspace(-5, 5, 10)
    pen_ratios = np.linspace(0, 1, num=6)
    
    # Cs = np.logspace(-7, 7, 30)
    # pen_ratios = np.linspace(0, 1, num=21)
    
    LR_params_C, LR_params_l1_ratio, to_save = run_gridsearch(X_val, y_val, Cs, cv=ps, pen_ratios= pen_ratios)
    
    LR_model.set_params(**LR_params_C)
    LR_model.set_params(**LR_params_l1_ratio)

    return LR_model, to_save








GAMMA = config.GAMMA
features = natsorted(config.features_list)



from glob import glob
features_list = glob('../data_backUp/feature_dataset/best_reduced_LR/*')
features = natsorted(features_list)





B = config.B
N_FOLDS = config.N_FOLDS
# splits_dir = '../../../param_optimization/data/splits/no_val/feature_datasets_splits_5Fold/'
# splits_dir = config.splits_dir_bootstrap
splits_dir = '../data_backUp/LR_classifier/splits/val_size=0.1/'



for feature in features:
    print('feature ', feature)
    
    """
    Read in data and drop unnecessary columns
    """
    df = pd.read_csv(feature, sep=';').dropna(thresh=2)
    df = df.drop(['ZCR', 'Kurtosis', 'LogE'], axis=1)
    feat_names_all = list(df.columns.drop(["Study_Num", "TB_status", 'Bin_No', 'Cough_No']))
    if 'Win_No' in feat_names_all:
        feat_names_all.remove('Win_No')
    
    
    
    fname_CSV = '../data_backUp/LR_classifier/results/feat_drop_SFS/' + feature.split('/')[-1].split('.')[0]+'.csv'
    # # Read from the CSV file if it is there 
    # results_df = pd.read_csv(fname_CSV, sep=';')
    
    ## This is to debug (DELETE LATER)
    # feat_names_all = feat_names_all[0:5]
    # feat_names_all = ['MFCC_3_mc', 'MFCC_11_mc', 'MFCC_D_14_mc', 'MFCC_12_mc', 'MFCC_5_mc']
    
    n_features = len(feat_names_all)
    # results_df, top_combinations_list = feat_drop_out(df, feat_names_all, n_features, save=False)
    
    
    
    """
    Recursively train on smaller set of features
    by dropping the worst performing groups

    :param df: full feature dataframe
    :param feat_names_all: list of feature names
    :param n_features: number of features to use (mostly for testing)
    :param n_group: number of filterbanks in a training group
    :param save: Flag for saving results to disk
    :return: results_df
    """
    
    # cut feat_names_all if n_features is given
    feat_names_all = feat_names_all[:n_features]
    
    step_1_save_name = '../data_backUp/LR_classifier/results/feat_drop_SFS/step_results/' + feature.split('/')[-1].split('.csv')[0] + '_step-1'
    if os.path.isfile(step_1_save_name):
        print('Step 1 has already been done for: ', feature)
        ## Load the data 
        with open(step_1_save_name, 'rb') as fp:
            feat_scores_list = pickle.load(fp)
    else:
        """
        Get score for each feature
        """
        print ('\n\n**** Step 1: Getting individual scores ... ')
        feat_scores_list = []
        for k in tqdm(range(n_features)):
            feats = feat_names_all[k:k+1]
            # print('\nSelected feature: ', feats)
            step_AUC_score = find_AUC(df, feats)
            print('\nAUC: ', step_AUC_score, ' for feature: ', feats)
            # b_score, b_err = get_bootstrap_auc(df, feature, feats)
            feat_scores_list.append((feats, step_AUC_score))
            # feat_scores_list.append((feats, b_score, b_err))
        print ('End of Step 1 ****\n\n')
        
        ## Save the data
        with open(step_1_save_name, 'wb') as fp:
            pickle.dump(feat_scores_list, fp, protocol=4)
    
    
    """
    Recursively train on top n features:
    
    Use best feature, and combine with all other features
    to find the best 2-combination
    
    Then use the best 2-combination and combine with all other
    features to find the best 3-combination
    
    etc etc..
    """
    
    fname_TXT = '../data/LR_classifier/results/feat_drop_SFS/' + feature.split('/')[-1].split('.')[0]+'.txt'
    
    
    
    
    best = max(feat_scores_list, key=itemgetter(1))
    
    error = np.std(list(np.float_(list(np.array(feat_scores_list)[:, 1]))))
    
    # Sort the features scores list by b_auc
    best_feats = best[0]
    # remove the best feature name from all feature names
    feat_names_all.remove(best_feats[0])
    
    print('\nBest AUC: ', best[1], ' for feature: ', best_feats)
    
    # List that will contain the top scoring combinations
    # Order is: (n_features (combinations), b_auc, b_err)
    top_scores_list = [ (1, best[1], error) ]
    # top_scores_list = [ (1, best[1]) ]
    top_combinations_list = []
    top_combinations_list.append(best_feats)
    
    with open(fname_TXT, 'a') as filehandle:
        # for listitem in top_combinations_list:
        filehandle.write('%s\n' % best_feats)
    
    
    print ('\n\n********* Step 2: Now finding the best combination ... ')
    for n_comb in tqdm(range(2, n_features+1)):
        print ('Number of features combined:', n_comb)
        
        step_2_save_name = '../data/LR_classifier/results/feat_drop_SFS/step_results/' + feature.split('/')[-1].split('.csv')[0] + '_step-2_comb-'+ str(n_comb)
        if os.path.isfile(step_2_save_name):
            print('Step 2 has already been done for: ', feature)
            ## Load the data 
            with open(step_2_save_name, 'rb') as fp:
                (top_scores_list, top_combinations_list) = pickle.load(fp)
            
            best_feats = top_combinations_list[-1]
            
            for f_n in best_feats:
                if f_n in feat_names_all:
                    feat_names_all.remove(f_n)
            
            
            # [feat_names_all.remove(f_n) for f_n in best_feats]
            
        else:
            
            if n_comb == n_features:
                ## Here, the top-scores should have the final all features 
                AUC_score = find_AUC(df, best_feats + feat_names_all)
                top_scores_list.append((n_features, AUC_score, 0))
                # top_scores_list.append((n_features, AUC_score))
                
                with open(fname_TXT, 'a') as filehandle:
                    filehandle.write('%s\n' % (best_feats + feat_names_all))
                
            else:
                # Combine best_feats_
                comb_scores = []
                for f in feat_names_all:
                    # Add this feature to the best combination list
                    feat_names = np.ravel(best_feats+[f])
                    # Get the scores
                    step_AUC_score = find_AUC(df, feat_names)
                    # b_score, b_err = get_bootstrap_auc(df, feature, feat_names)
                    # Save the scores
                    comb_scores.append((f, step_AUC_score))
                    # comb_scores.append((f, b_score, b_err))
                    # print feat_names, b_score
                
                # Get the feature that gave the best score when added to best_feats
                best_combination = max(comb_scores, key=itemgetter(1))
                
                # np.std(list(np.float_(list(np.array(comb_scores)[:, 1]))))
                
                error = np.std(list(np.float_(np.array(comb_scores)[:, 1])))
                
                
                
                # print('\nAUC: ', best_combination[1], ' for feature combination: ', best_feats)
                feat_max = best_combination[0]
                
                print('\nAUC: ', best_combination[1], ' for feature combination: ', best_feats[0]+', '+feat_max)
                
                # Remove the feature that gave the best score
                feat_names_all.remove(feat_max)
        
                top_combinations_list.append(best_feats)
                
                best_feats.append(feat_max)
                
                with open(fname_TXT, 'a') as filehandle:
                    # for listitem in best_feats:
                    filehandle.write('%s\n' % best_feats)
                
                # print (top_combinations_list)
        
                # Save: (number of features combined, boot auc of combination, boot err of combination)
                top_scores_list.append((n_comb, best_combination[1], error))
            
            ## Save the data
            with open(step_2_save_name, 'wb') as fp:
                pickle.dump((top_scores_list, top_combinations_list), fp, protocol=4)
    
    
    print ('********* End of Step 2 *********\n\n')
    
    # top_scores_list.insert(0, (0, 0))
    top_scores_list.insert(0, (0, 0, float('NaN')))
    
    
    results_df = pd.DataFrame(np.vstack(top_scores_list), columns=['No.Features', 'AUC', 'Error'])
    # results_df = pd.DataFrame(np.vstack(top_scores_list), columns=['No.Features', 'AUC'])
    
    
    # yerr = []
    # for i in results_df['AUC']:
    #     yerr.append(i*0.05)
    
    # results_df = pd.read_csv(fname_CSV, sep=';')
    
    yerr=results_df['Error']
    
    yerr_new = []
    for ye_index in range(len(yerr)):
        if ye_index%5 == 1:
            yerr_new.append(yerr[ye_index])
        else:
            yerr_new.append(float('NaN'))
    
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.errorbar(x=results_df['No.Features'], y=results_df['AUC'],
                yerr=yerr_new, ecolor='red',
                  capsize=10, color='green', 
                  label='ROC curve with error bars, shown for every 5 combinations')
    plt.legend(loc='lower right', fontsize=20)
    # plt.plot([0, n_features], [0, 1], color='navy', linestyle='--')
    
    # plt.errorbar(x=results_df['No.Features'], y=results_df['AUC'],
                # yerr=yerr_new, ecolor='red',
                  # capsize=10, color='green')
    
    # plt.errorbar(x=results_df['No.Features'], y=results_df['AUC'],
    #             yerr=yerr, ecolor='red',
    #               capsize=10, color='green')
    # plt.errorbar(x=results_df['No.Features'], y=results_df['AUC'],
    #             yerr=yerr, capsize=5, capthick=2,
    #             linestyle='-', marker='.',ecolor='darkred',
                # errorevery=5)
    
    plt.xlim(0, n_features+1)
    plt.ylim(0, 1.02)
    plt.xlabel('Number of features', fontsize=20)
    plt.ylabel('Area under curve (AUC)', fontsize=20)
    
    plt.xticks(np.arange(0, n_features, 5), fontsize=15)
    yTick_list = []
    for n in np.linspace(0, 1, 11):
        yTick_list.append(str(int(n*100))+'%')
    plt.yticks(np.linspace(0, 1, 11), yTick_list, fontsize=15)
    # plt.title('Sequential Forward Search', fontsize=35)
    plt.grid(color='y', linewidth=0.5)
    plt.show()
    
    
    fname_PNG = '../data/LR_classifier/results/feat_drop_SFS/' + feature.split('/')[-1].split('.')[0]+'.png'
    plt.savefig(fname_PNG)
    plt.close()
    
    
    # fname_CSV = '../data/LR_classifier/results/feat_drop_SFS/' + feature.split('/')[-1].split('.')[0]+'.csv'
    results_df.to_csv(fname_CSV, sep=';', index=False)
    
    print ('Saving feature combinations to disk...')
    
    


