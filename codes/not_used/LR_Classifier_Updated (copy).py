#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 21:45:31 2020

@author: madhu
"""

import os
import csv
import sys
import random
import datetime
import numpy as np
import pandas as pd

from operator import mul
from scipy.stats.mstats import gmean

from glob import glob
from helper import *
from natsort import natsorted
from operator import itemgetter

from sklearn.metrics import *
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.preprocessing import binarize, scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, PredefinedSplit, GridSearchCV
# from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt


def exit():
    sys.exit()

def ROC_analysis(prob_ref_list, thresh_ = True, plot_=False, fname = None, name = ''):

    """
    :param prob_ref_list    List of tuples in ZIP format (prob, ref_label)
    :param plot:            Flag for plotting ROC curve
    :param thresh_          Flag to return equal error threshold
    :param fname            Name of feature file - make dir with that name
    :param name             Name to be put in graph title. Name of eval method
    :return:                optimized predictions (opt_pred)
                            area under the curve (AUC_ACC)
    """
    
    [probs, y_true] = zip(*prob_ref_list)
    # roc_curve: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    # Compute Receiver operating characteristic (ROC)
    fpr, tpr, thresholds = roc_curve(y_true, probs, drop_intermediate=False)
    # auc: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
    # Compute Area Under the Curve (AUC) using the trapezoidal rule
    auc_acc = auc(fpr, tpr)
    
    i = np.arange(len(tpr))
    # Sensitivity = true positive rate = tpr
    # Specificity = true negetive rate (1-false positive rate) = 1 - fpr
    # equal-error rate = Sensitivity - Specificity = (tpr - (1 - fpr)
    roc = pd.DataFrame({'fpr': pd.Series(fpr, index=i),
                        'tpr': pd.Series(tpr, index=i),
                        '1-fpr': pd.Series(1 - fpr, index=i),
                        'tf': pd.Series(tpr - (1 - fpr), index=i),
                        'thresholds': pd.Series(thresholds, index=i)
                        })
    # Find out the index number where the 'equal-error rate' is minimum 
    idx = (roc.tf).abs().argmin()
    # This threshold is the GAMMA_EE mentioned in Renier's thesis, page 84. 
    threshold = roc.thresholds.iloc[idx]

    # Do diagnosis with optimized threshold
    opt_preds = map(int, binarize(np.array(probs).reshape(1, -1), threshold=threshold)[0])
    
    if plot_:
        fname = os.path.basename(fname)[:-4]
        fig_out_dir = '/home/madhu/work/cough_classification/data/LR_classifier/results/ROC_figs/LR_Classifier/{}'.format(fname)
        if not os.path.isdir(fig_out_dir):
            os.mkdir(fig_out_dir)
        plt.figure()
        plt.plot(fpr, tpr, color = 'darkorange',label = 'ROC curve (AUC = {})'.format(auc_acc))
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1], color = 'navy', linestyle='--')
        plt.xlim([-0.05,1.0])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for {}'.format(name))
        plt.savefig(fig_out_dir+'/ROC_{}.png'.format(name))
        plt.close()
        
    if thresh_:
        return opt_preds, auc_acc, threshold

    else:
        return opt_preds, auc_acc


def timeStamped(fname, dir_=False):
    if dir_:
        fmt = '{fname}_%m-%d-%Hh-%Mm/'
    else:
        fmt = '{fname}_%m-%d-%Hh-%Mm.csv'

    return datetime.datetime.now().strftime(fmt).format(fname=fname)


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

        labels = np.array([df[df.Study_Num == rec].TB_Result.values[0] for rec in recs])

        # skf = StratifiedKFold(labels, n_splits=n_folds)
        skf = StratifiedKFold(n_splits=n_folds)
        
        y_ps = np.zeros(shape=(len(df.index),))
        fold = 0
        for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
            
        
        # y_ps = np.zeros(shape=(len(df.index),))

        
        # for train_idx, test_idx in skf:
            test_recs = recs[test_idx]

            ps_test_idx = np.array(df[df['Study_Num'].isin(test_recs)].index)

            y_ps[ps_test_idx] = fold

            fold += 1

        return PredefinedSplit(y_ps)

    def run_gridsearch(X, y, model, params, cv):

        if CV_SCORER == 'roc_auc':
            # Make Cohen's Kappa Scorer
            # kappa_scorer = make_scorer(cohen_kappa_score)
            # Initiate GS
            grid_search = GridSearchCV(model, param_grid=params, cv=cv, scoring=CV_SCORER)
            # print('The scoring method used is roc_auc')
        else:
            # Initiate GS
            grid_search = GridSearchCV(model, param_grid=params, cv=cv)
        
        # Scale the data
        X = scale(X)
        # y = scale(y)
        
        # print('X', X)
        # print('y', y)
        # print('shape of y: ', y.shape)
        # Fit GS with this data
        grid_search.fit(X, y)
        
        # # Get the scores, sorted from best to worst
        # scores = sorted(grid_search.grid_scores_, key=itemgetter(1), reverse=True)

        # # Take the params of the best scores
        # gs_opt_params = scores[0].parameters

        # print (scores)
        
        gs_opt_params = grid_search.best_params_
        
        print (gs_opt_params)
        # exit()

        return gs_opt_params

    df_ = df.copy()
    # Reset index for referencing
    df_.reset_index(inplace=True)

    # Make the PredefinedSplit label
    ps = make_ps(df_, n_folds=2)
    
    # labels
    y_val = df_.TB_Result.values
    # data
    X_val = df_[feat_names]

    LR_model = LogisticRegression(max_iter=100000)

    LR_param_grid = {'C': np.logspace(-5, 5, 10)}
    # print('ps: ', ps)
    LR_params = run_gridsearch(X_val, y_val, LR_model, LR_param_grid, cv=ps)

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
    dev_labels = np.array([dev_df[dev_df.Study_Num == rec].TB_Result.values[0] for rec in dev_recs])

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

        y_train = list(train_df.TB_Result.values)
        y_test = list(test_df.TB_Result.values)

        X_train = train_df[feat_names]
        X_test = test_df[feat_names]

        # Train the LRM
        LRM.fit(X_train, y_train)

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
    y = list(data.TB_Result.values)
    X = data[feat_names]
    LRM.fit(X, y)

    return LRM


def test_model(LRM, test_data):

    def TBI_eval(probs, y_ref, test_px, cough_nums):

        # Get the different TBI probs and refs
        i = np.arange(len(test_px))

        df = pd.DataFrame({"Recording": pd.Series(test_px, index=i),
                           "Reference": pd.Series(y_ref, index=i),
                           "Probabilities": pd.Series(probs, index=i),
                           'Cough_No': pd.Series(cough_nums, index=i)
                           })
        
        tbi_ref = [] # Recording based labels
        TBI_a_list = [] # Arithmetic mean of all probabilities of all cough in one recording
        TBI_g_list = [] # Geometric mean of all probabilities of all cough in one recording
        TBI_s_list = [] # Ratio of pos_coughs/all_coughs in recording

        for name, group in df.groupby("Recording"):
            
            # print(name)
            # print(group)
            
            l = group.Reference.iloc[0]
            tbi_ref.append(l)

            # TBI_A
            TB_a_prob = sum(group.Probabilities.values) / float(len(group.Probabilities))
            TBI_a_list.append(TB_a_prob)

            # TBI_G
            TB_g_prob = np.exp(1.0 / len(group.Probabilities.values) * sum(np.log(group.Probabilities.values)))
            TBI_g_list.append(TB_g_prob)

            # TBI_S
            # This is the TIS as explained in page number 86 of Renier's thesis. 
            pos_coughs_count = 0
            all_coughs_count = 0
            # Count all coughs in recording
            for cough_no, cough_df in group.groupby('Cough_No'):
                # print(cough_no)
                # Probability that this cough is TB+
                cough_prob = sum(cough_df.Probabilities.values) / float(len(cough_df.Probabilities))
                if cough_prob >= GAMMA:
                    pos_coughs_count += 1
                all_coughs_count += 1
            # Save ratio of TB+ coughs/all coughs
            TBI_s_list.append(float(pos_coughs_count) / all_coughs_count)

        return [TBI_a_list, TBI_g_list, TBI_s_list], tbi_ref

    # Get the study numbers / patients, TB_results, features etc. 
    test_recs = list(test_data.Study_Num.values)
    cough_nums = list(test_data.Cough_No.values)
    test_ref = list(test_data.TB_Result.values)
    X_test = test_data[feat_names]
    
    # Predict the probabilities of the test set, this should match the outputs with LRM.predict(X_test)
    test_probs = LRM.predict_proba(X_test)[:, 1]

    """
    Convert per-frame probs to per-recording probs
    TBI_probs = [TBI_A_probs, TBI_G_probs, TBI_S_probs]
    """
    TBI_probs, TBI_ref = TBI_eval(probs = test_probs,
                                  y_ref = test_ref,
                                  test_px = test_recs,
                                  cough_nums = cough_nums)


    test_ = zip(test_probs, test_ref)

    return test_, TBI_probs, TBI_ref


def write_log(f_results_out):
    """
	Write a small note about the nature of these results
	"""

    log = {'SPLITS_DIR': splits_dir,
           'FEATURES_DIR': features_dir,
           'VAL_SIZE %': val_size * 100,
           'TBI_METHOD': 'BOTH',
           'CV_SCORER': CV_SCORER}

    f_log = log_out_dir_name + os.path.basename(f_results_out).split('.')[0] + '_LOG.csv'

    # Write the log file to disk
    with open(f_log, 'wb') as log_csv:
        writer = csv.writer(log_csv, delimiter=';')
        for key, val in log.items():
            # print('key', key)
            # print('val', val)
            writer.writerow([key, val])


def write_out_results(d, val_scores, TBI_scores, f_out,eval_type):

    """
    All scores except val,are in the shape (3,5)
    Each row represents A/G/S TBI type
    Columns are ordered: [sens, spec, acc, auc, kappa]

    """

    # Manipulate f_out
    temp_dir = os.path.join(os.path.split(f_out)[0],eval_type)

    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)

    f_out = temp_dir + '/' + os.path.basename(f_out)

    columns = ['CONFIG_N', 'CONFIG_M', 'CONFIG_B', 'CONFIG_AVG']
    columns.extend(['VAL_Sens', 'VAL_Spec', 'VAL_Acc', 'VAL_AUC', 'VAL_KAPPA'])
    columns.extend(['TBI_A_Sens', 'TBI_A_Spec', 'TBI_A_Acc','TBI_A.AUC', 'TBI_A_Kappa'])
    columns.extend(['TBI_G_Sens', 'TBI_G_Spec', 'TBI_G_Acc','TBI_G.AUC', 'TBI_G_Kappa'])
    columns.extend(['TBI_S_Sens', 'TBI_S_Spec', 'TBI_S_Acc','TBI_S.AUC', 'TBI_S_Kappa'])

    results_df = pd.DataFrame(columns=columns)

    # Put the config data in the right format for dataframe
    # cfg_data = [d['N'], d['M'], d['B'], d['Avg']]
    # cfg_data = [ 256, 256, 1, False]
    
    N = d.split('M=')[-1].split('_B')[0]
    M = d.split('M=')[-1].split('_B')[0]
    B = d.split('B=')[-1].split('_Avg')[0]
    Avg = d.split('Avg=')[-1]
    
    cfg_data = [ N, M, B, Avg]
    
    # Start the results data by adding cfg data
    results_data = cfg_data

    # Add validation scores
    results_data.extend(val_scores)

    # Add TBI_A scores
    results_data.extend(TBI_scores[0])
    # Add TBI_G scores
    results_data.extend(TBI_scores[1])
    # Add TBI_S scores
    results_data.extend(TBI_scores[2])

    # Add the data to dataframe
    # For this version, just make a single line
    results_df.loc[0] = results_data

    # If file doesn't exist, create and initialize format
    if not os.path.isfile(f_out):
        results_df.to_csv(f_out, sep=';', index=False, mode='w')

    # Append to existing file
    else:
        results_df.to_csv(f_out, sep=';', index=False, mode='a', header=False)


def get_f_results_out(features_dir):
    """
	Generate the results output name
	from the features directory name
	"""
    if 'N_FBANKS' in features_dir:
        temp = os.path.basename(os.path.normpath(features_dir))
        N_FBANKS = int(temp.split('=')[1].split('_')[0])
        # N_FBANKS = int(temp.split('FBANKS=')[1].split('.')[0])
        spec = temp.split('=')[-1]
        f_results_out = ''.join([out_dir_name, "N_FBANKS=", str(N_FBANKS), "_SPEC=", spec, '.csv'])
        N_MFCC = None

    elif 'N_MFCC' in features_dir:
        N_MFCC = int(os.path.basename(os.path.normpath(features_dir)).split('=')[1])
        f_results_out = ''.join([out_dir_name, "N_MFCC=", str(N_MFCC), '.csv'])
        N_FBANKS = None
        spec = None

    else:
        print ('DIR MAYBE NOT RIGHT:')
        print (features_dir)
        exit()

    return f_results_out, N_MFCC, N_FBANKS, spec


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
    F = 2 * (SPEC*SENS) / (SPEC+SENS)

    KAPPA = calc_kappa(y_ref, preds)

    return [SENS, SPEC, ACC, KAPPA]


def save_probs(val_list, test_list, tbi_probs_arr,tbi_ref_list, fname):


    def write_prob_to_disk(output_dir, df, fname):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        probs_df.to_csv('{}/{}'.format(output_dir, os.path.basename(fname)),index=False, sep=';')


    # Save val probs
    output_dir = '{}/{}'.format(probs_out_dir,'val')
    probs_df = pd.DataFrame(np.array(val_list), columns = ['Probability','Reference'])
    write_prob_to_disk(output_dir, probs_df, fname)


    # Save test_probs
    output_dir = '{}/{}'.format(probs_out_dir,'test')
    probs_df = pd.DataFrame(np.array(test_list), columns = ['Probability','Reference'])
    write_prob_to_disk(output_dir, probs_df, fname)

    # Save TBI_A probs
    output_dir = '{}/{}'.format(probs_out_dir,'TBI_A')
    probs_df = pd.DataFrame(list(zip(tbi_probs_arr[0],tbi_ref_list)), columns = ['Probability','Reference'])
    write_prob_to_disk(output_dir, probs_df, fname)

    # Save TBI_A probs
    output_dir = '{}/{}'.format(probs_out_dir,'TBI_G')
    probs_df = pd.DataFrame(list(zip(tbi_probs_arr[1],tbi_ref_list)), columns = ['Probability','Reference'])
    write_prob_to_disk(output_dir, probs_df, fname)

    # Save TBI_A probs
    output_dir = '{}/{}'.format(probs_out_dir,'TBI_S')
    probs_df = pd.DataFrame(list(zip(tbi_probs_arr[2],tbi_ref_list)), columns = ['Probability','Reference'])
    write_prob_to_disk(output_dir, probs_df, fname)



val_size = 0.25
GAMMA = 0.5

config_dir = '/home/madhu/work/cough_classification/data/LR_classifier/cfg/'
splits_dir = '/home/madhu/work/cough_classification/data/LR_classifier/splits/val_size=' + str(val_size)+'/'
# splits_dir += '/feature_datasets_splits/'

# CV_SCORER = 'DEFAULT'
CV_SCORER = 'roc_auc'

# This is the number of folds which should be the same with the number of splits. 
N_Fold = 10
    
# Make output dir:
out_dir_name = '/home/madhu/work/cough_classification/data/LR_classifier/results/single_classifier/eval_comparison/'
# os.mkdir(out_dir_name)

# Make log dir:
log_out_dir_name = '/home/madhu/work/cough_classification/data/LR_classifier/results/single_classifier/logs/'
# os.mkdir(log_out_dir_name)

probs_out_dir = '/home/madhu/work/cough_classification/data/LR_classifier/results/single_classifier/probabilities/'
# os.mkdir(probs_out_dir)

# features_dir_list = natsorted(glob('../data/feature_datasets/O_DROP/*'))
features_dir_list = natsorted(glob('/home/madhu/work/cough_classification/data/LR_classifier/features_dataset/*'))

for features_dir in features_dir_list:
    
    f_results_out = '/home/madhu/work/cough_classification/data/LR_classifier/results/'+features_dir.split('/')[-1]
    
    # N_FBANKS = 80
    # N_MFCC = 13
    
    print ("\n\n\nFEATURES_DIR:", os.path.basename(features_dir))
    print ("SPLITS_DIR = ", splits_dir)
    print ("RESULTS_FNAME = ", f_results_out)
    
    # write_log(f_results_out)
    
    # config_files = natsorted(glob(config_dir + "/*.config"))
    # count = 0
    # for cfg in config_files:

        # d_cfg = load_conf_dict(cfg)

        # feat_csv = make_fname_from_dict(d_cfg, features_dir, N_FBANKS=N_FBANKS, spec=spec, N_MFCC=N_MFCC)

    print ("Loading data from ", os.path.basename(features_dir))
    # feat_csv = features_dir
    try:
        data = pd.read_csv(features_dir)
    except:
        print('\n\n\nfile does not exist.\n\n\n')
        continue

    """
			Drop any NaNs. Set thresh to 2 because 
			Win_No is NaN for Avg=True datasets
			"""
    data = data.dropna(thresh=2)
    #  column names
    feat_names = list(data.columns.drop(["Study_Num", "TB_Result", 'Cough_No']))

    """
			Need to add this, I messed up with the Win_No column
			"""
    if "Win_No" in feat_names:
        feat_names.remove("Win_No")
    
    # Frame-based
    val_list    = []
    val_scores  = [] 
    test_list   = []

    # Rec-based
    tbi_probs_list  = []
    # references
    tbi_ref_list    = []
    # Predictions
    tbi_preds_list = []
    # results
    tbi_scores_roc_list = []
    tbi_scores_gamma_list = []
    
    # For each split
    for k in range(1, N_Fold+1):
        print ('fold no: ', k)
        train_df, test_df, val_df = load_splits(full_df=data, f=features_dir, k=k)
        
        # print('Train patient: ', list(np.unique(np.array(train_df.Study_Num))), '\n')
        # print('Test patient: ', list(np.unique(np.array(test_df.Study_Num))), '\n')
        # print('Validation patient: ', list(np.unique(np.array(val_df.Study_Num))), '\n')
        
        # Combine train and val sets to create dev set
        dev_df = train_df.append(val_df).sort_values('Study_Num')
        
        """
        Get Optimal LR model using validation dataset
        """
        opt_LRM = cv_param_estimation(val_df)
        
        
        """
        Get validation results and Equal-Error-Rate
        threshold GAMMA

        Also get validation scores
        """
        
        # Get probabilities and reference in the 'zip' format 
        val_pr = validation(opt_LRM, dev_df)
        [probs, y_true] = zip(*val_pr)
        # Evaluate to get GAMMA
        val_predictions, val_auc, GAMMA = ROC_analysis(zip(probs, y_true), thresh_=True)
        # Get: SENS, SPEC, ACC, KAPPA 
        val_score = evaluate_model(y_true, val_predictions)
        # Add area-under-curve at 4th position: SENS, SPEC, ACC, AUC, KAPPA 
        val_score.insert(3,val_auc)
        
        """
        Train model on entire dev set
        """
        trained_LRM = train_model(opt_LRM, dev_df)
        
        # Test model
        test_pr, TBI_probs, TBI_ref = test_model(trained_LRM, test_df)
        
        """
        Make predictions using GAMMA & 0.5 (S)
        """
        TBI_A_preds = map(int, binarize(np.array(TBI_probs[0]).reshape(1, -1), threshold=GAMMA)[0])
        TBI_G_preds = map(int, binarize(np.array(TBI_probs[1]).reshape(1, -1), threshold=GAMMA)[0])
        TBI_S_preds = map(int, binarize(np.array(TBI_probs[2]).reshape(1, -1), threshold=0.5)[0])
        TBI_preds = [TBI_A_preds, TBI_G_preds, TBI_S_preds]
        
        
        
        """
        Make predictions using ROC analysis
        and finding the EER.

        Evaluate with ROC predictions and
        GAMMA predictions
        """
        
        TBI_scores_roc = []
        TBI_scores_gamma = []
        for i in range(3):
            # Evaluate ROC
            predictions, auc_acc = ROC_analysis(zip(TBI_probs[i], TBI_ref), thresh_ = False)
            scores = evaluate_model(TBI_ref, predictions)
            scores.insert(3,auc_acc)
            TBI_scores_roc.append(scores)
            
            # Evaluate GAMMA (use the same auc_acc)
            scores = evaluate_model(TBI_ref, TBI_preds[i])
            scores.insert(3,auc_acc)
            TBI_scores_gamma.append(scores)


        # Save all the probabilities and reference labels
        val_pr = validation(opt_LRM, dev_df)
        val_list.extend(val_pr)
        val_scores.append(val_score)

        test_list.extend(test_pr)
        tbi_probs_list.append(TBI_probs)
        tbi_ref_list.extend(TBI_ref)

        # Gamma preds
        tbi_preds_list.append(TBI_preds)
        # ROC scores
        tbi_scores_roc_list.append(TBI_scores_roc)
        tbi_scores_gamma_list.append(TBI_scores_gamma)
        
    
    """
    Predictions and probs in array with shape (3,)
    row 0 : TBI_A
    row 1 : TBI_G
    row 2 : TBI_S
    Scores in array of shape (10,3,[])
    10 - folds
    3 - A,G,S
    [] - [sens, spec, acc]
    """
    val_score_means      = np.mean(np.array(val_scores),axis=0)
    tbi_probs_arr       = np.hstack(tbi_probs_list)
    tbi_scores_gamma    = np.mean(np.array(tbi_scores_gamma_list),axis=0)
    tbi_scores_roc      = np.mean(np.array(tbi_scores_roc_list),axis=0)

    """
    Save the probabilities and references to disk
    """
    save_probs(val_list, test_list, tbi_probs_arr, tbi_ref_list, features_dir)


    """
    Evaluate using probabilities over all folds
    """
    tbi_scores_overall_roc = np.empty(shape=(3,5))
    for x,name in zip([0,1,2],['TBI_A','TBI_G','TBI_S']):

        # Get overall AUC, and at the same time do ROC overall and save graph
        predictions, auc_acc = ROC_analysis(zip(tbi_probs_arr[x],tbi_ref_list), 
                                                thresh_ = False,
                                                plot_ = True,
                                                fname = features_dir,
                                                name = name)
        # Now evaluate overall_roc predictions
        scores = evaluate_model(tbi_ref_list, predictions)
        scores.insert(3,auc_acc)
        tbi_scores_overall_roc[x] = scores


    d_cfg = features_dir.split('/')[-1].split('.')[0]
    write_out_results(d_cfg, val_score_means,tbi_scores_gamma, f_results_out,eval_type='GAMMA')
    write_out_results(d_cfg, val_score_means,tbi_scores_roc, f_results_out,eval_type='ROC')
    write_out_results(d_cfg, val_score_means,tbi_scores_overall_roc, f_results_out,eval_type='OVERALL_ROC')

    # count += 1

