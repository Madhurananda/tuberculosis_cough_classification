#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:12:22 2020

@author: madhu
"""

import os
import csv
import sys
import time
import random
import datetime
import numpy as np
import pandas as pd

from glob import glob

from natsort import natsorted
from numpy import interp

from sklearn.metrics import *
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.preprocessing import binarize, scale
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, PredefinedSplit

import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, './Renier_scripts')
from helper import *

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

    train_df = full_df[full_df.Study_Num.isin(train_recs)]
    test_df = full_df[full_df.Study_Num.isin(test_recs)]

    return train_df, test_df


def cv_param_estimation(df):
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

    def run_gridsearch(X, y, Cs, cv):
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
            
            grid_search = LogisticRegressionCV(Cs=Cs, cv=cv, scoring='roc_auc', max_iter = 100000)
            
        else:
            grid_search = LogisticRegressionCV(Cs=Cs, cv=cv)
        
        # Scale the data
        X_scaled = scale(X)
        
        # from sklearn.preprocessing import robust_scale
        # X_r_scaled = robust_scale(X)
        
        # Fit GS with this data
        grid_search.fit(X_scaled, y)

        gs_opt_params = {'C': grid_search.C_[0]}
        
        # gs_opt_params = grid_search.best_params_
        
        print (gs_opt_params)

        return gs_opt_params

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

    LR_model = LogisticRegression(max_iter=100000)

    Cs = np.logspace(-5, 5, 10)

    LR_params = run_gridsearch(X_val, y_val, Cs, cv=ps)

    LR_model.set_params(**LR_params)

    return LR_model


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






GAMMA = 0.5
N_FOLDS = 15
CV_SCORER = 'AUC'

output_File = '/home/madhu/work/cough_classification/data/LR_classifier/results/output.txt'
# splits_dir = '/home/madhu/work/cough_classification/data/LR_classifier/splits/val_size=0.25/'

feat_dir = '/home/madhu/work/cough_classification/data/LR_classifier/features_dataset/'
features = os.listdir(feat_dir)
features = natsorted(features)

# features = ['features_MFCC=39_Frame=2048_B=1_Avg=False.csv']
# features = ['features_FBANK=100_Frame=512_B=1_Avg=True.csv']
# features = ['features_MFCC=13_Frame=2048_B=1_Avg=True.csv']
features = ['features_MFCC=13_Frame=2048_B=1_Avg=True.csv']


val_sizes = [0.1, 0.15, 0.2, 0.25]
# val_sizes = [0.15, 0.2, 0.25]

for val_size in val_sizes: 
    
    # split_dataset(splits_dir, natsorted(feat_list))
    splits_dir = '/home/madhu/work/cough_classification/data/LR_classifier/splits/'
    splits_dir += 'val_size='+ str(val_size)+'/'
    print(splits_dir)
    
    # splits_dir = '/home/madhu/work/cough_classification/data/LR_classifier/splits/val_size=0.25/'
    
    for feature in features:
        
        saveFig_title = '/home/madhu/work/cough_classification/data/LR_classifier/results/ROC_figs/LR_Classifier/'+str(feature[0:len(feature)-4])+'_split='+str(val_size)+'.png'
        
        # First check if the classification has already done or not
        if os.path.isfile(saveFig_title):
            print('Analysis has already been done for: ', feature, ' split: ', val_size)
        else:
            
            print('Running LR classifier on: ', feature, ' split: ', val_size)
            
            data = pd.read_csv(feat_dir+feature, sep=';')
            
            """
            	Drop any NaNs. Set thresh to 2 because
            	Win_No is NaN for Avg=True datasets
            	"""
            data = data.dropna(thresh=2)
            
            data.TB_status = data.TB_status.astype(float)
            
            #  column names
            feat_names = list(data.columns.drop(["Study_Num", "TB_status", 'Bin_No', 'Cough_No']))
            
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
            
            # For each split
            for k in range(1, N_FOLDS + 1):
                print ("FOLD:", k)
                train_df, test_df = load_splits(full_df=data, f=feature, k=k)
            
                """
                Get Optimal LR model using validation dataset
                """
                try:
                    opt_LRM = cv_param_estimation(train_df)
                except:
                    now = datetime.now()
                    current_time = now.strftime("%Y/%m/%d at %H:%M:%S")
                    to_save = '\n\nTime Now: ' + str(current_time) + ' Feature: ' + str(feature) + ' Split: ' + str(val_size)
                    to_save += '\nError in LR Classifier. \n\n'
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
                    
                    # Evaluate using GAMMA
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
            
                # Evaluate using GAMMA
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
            
            # Make the plot and then save it
            plt.plot(mean_fpr, mean_tpr_ADS, color='darkorange', label='ADS ROC curve (AUC = {:.4f})'.format(mean_auc_ADS))
            plt.plot(mean_fpr, mean_tpr_TBI[1], color='green', label='TIS ROC curve (AUC = {:.4f})'.format(mean_auc_TIS))
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlim([-0.05, 1.0])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Mean ROC curve')
            plt.show()
            
            
            plt.savefig(saveFig_title)
            plt.close('all')
            
            
            
            now = datetime.now()
            current_time = now.strftime("%Y/%m/%d at %H:%M:%S")
            
            to_save = '\n\nTime Now: ' + str(current_time) + ' Feature: ' + str(feature) + ' Split: ' + str(val_size)
            to_save += '\nADS mean ROC: ' + '{:.4f}'.format(mean_auc_ADS)
            to_save += '\nTIS mean ROC: ' + '{:.4f}'.format(mean_auc_TIS)
            
            to_save += '\n\n'
            # Save the results only if ROC is greater than 0.75
            if mean_auc_ADS > 0.75 or mean_auc_TIS > 0.75:
                write_file(output_File, to_save)
                
            
            
            
            # Insert the mean auc values into the scores array
            tbi_scores_gamma = np.insert(tbi_scores_gamma, 3, mean_auc_TIS, axis=1)
        
        
        