#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 00:19:18 2020

@author: madhu
"""

"""
Investigate which frequency bands contain the most information

"""

import os
import numpy as np
import pandas as pd
import helper_stratified as help

from numpy import interp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns
import config

from sklearn.preprocessing import scale, robust_scale

B = config.B
N_FOLDS = config.N_FOLDS
SPLITS = 20
splits_dir = config.splits_dir_bootstrap
# splits_dir = '/home/madhu/work/cough_classification/Renier_data/Code/Experiments/param_optimization/data/splits/no_val/feature_datasets_splits_5Fold/'


def get_bootstrap_auc(df, test_csv, feat_names, B = 100):
    """
    Get the auc using bootstrap sampling

    :param df: Full dataframe
    :param feat_names: List of features
    :param B: Number of bootstrap samples
    :return: AUC_boot: Bootstrap AUC (mean over all bootstrap samples)
    :return: boot_se: Bootstrap standard error

    """

    def load_stratifiedSamples(k, dir):

        """
        Use the labels to perform random stratified sampling
        with replacement and return the recording of the sampled
        subset.
        :param k: Fold no
        :param dir: Bootstrap dir
        """
        f_bootstrap_sample = dir + 'FOLD_{}.csv'.format(k)

        bootstrap_sample_df = pd.read_csv(f_bootstrap_sample, sep=';', usecols=[1], header=None, names=['recs'])

        return [eval(r) for r in bootstrap_sample_df['recs'].values]

    # N = test_csv.split('.')[-2].split('=')[-1]
    N = test_csv.split('=')[1].split('_')[0]

    bootstrap_dir = '../data/feature_dataset/bootstrap_samples/N={}/B={}/'.format(N, B)

    mean_fpr = np.linspace(0, 1, 20)
    mean_tpr = [0.0]*B

    for k in range(1, N_FOLDS + 1):
        print ('Fold:',k)
        
        train_df, test_df = help.load_splits(splits_dir, df, test_csv, k)

        """
        Grid search to get best params
        """
        param_grid = {'C': np.logspace(-5, 5, 10)}
        clf = help.cv_param_estimation(train_df,
                                       LogisticRegression(max_iter=100000),
                                       param_grid,
                                       n_folds=4,
                                       feat_names=feat_names,
                                       type='LR')
        """
        Cross validation on train set
        """
        val_pr = help.validation(clf, train_df, feat_names)
        # Get equal error rate threshold
        GAMMA = help.get_gamma_ee(val_pr)

        """
        Train classifier
        """
        # clf.fit(train_df[feat_names], train_df.TB_status.values)
        X = train_df[feat_names]
        y = train_df.TB_status.values
        
        X_r_scaled = robust_scale(X)
        
        clf.fit(X_r_scaled, y)


        """
        Test the classifier using the whole test set
        """
        TBI_probs, TBI_ref, TBI_recs = help.test_model_b(clf, test_df, feat_names, GAMMA)

        idx_dict = dict((val, idx) for idx, val in enumerate(TBI_recs))

        """
        Test classifier using
        a bootstrap sample set
        """
        bootstrap_recs_list = load_stratifiedSamples(k, bootstrap_dir)

        b = 0
        # print 'b=',
        for bootstrap_recs in bootstrap_recs_list:
            # if b%1000 == 0:
            #     print b,
            """
            Ek kan nog later die testing buit die loop sit en net
            die TBI_probs slice wat in die bootstrap sample is
            """
            test_df_b = test_df[test_df.Study_Num.isin(bootstrap_recs)]

            # TBI_probs, TBI_ref = help.test_model(clf, test_df_b, feat_names, GAMMA)

            """
            Get probs of this bootstrap sample
            """
            boot_idx = [idx_dict[x] for x in np.unique(test_df_b.Study_Num.values)]
            boot_probs = np.array(TBI_probs[0])[boot_idx]
            boot_ref = np.array(TBI_ref)[boot_idx]


            # Skipping TIS results for now
            fpr, tpr, _ = roc_curve(boot_ref, boot_probs)
            mean_tpr[b] += interp(mean_fpr, fpr, tpr)
            mean_tpr[b][0] = 0.0
            b += 1


    # Get the AUC over all folds for each bootstrap sample
    b_auc = []
    for b in range(B):
        # Normalize the mean_tpr
        m_tpr = mean_tpr[b]
        m_tpr /= N_FOLDS
        m_tpr[-1] = 1.0

        # Get AUC
        m_auc = auc(mean_fpr, m_tpr)
        b_auc.append(m_auc)

    """
    AUC_boot - the mean AUC over all bootstrap samples
    """
    AUC_boot = np.mean(b_auc)


    """
    Bootstrap Standard error
    square root of the mean square error
    (sum of all bootstrap aucs - AUC_boot)**2 divided
    by B-1
    """
    b_se = np.sqrt((np.sum((np.array(b_auc)-AUC_boot)**2)) / (B-1))

    return AUC_boot, b_se


def get_auc(df, test_csv, feat_names):
    """
    Get the results considering all features

    :param df: Full dataframe
    :param feat_names: List of features
    :return: auc: AUC accuracy
    """

    mean_fpr = np.linspace(0, 1, 20)
    mean_tpr = [0.0, 0.0]

    # print 'Fold:',
    for k in range(1, N_FOLDS + 1):
        # print k,

        train_df, test_df = help.load_splits(splits_dir, df, test_csv, k)

        """
        Grid search to get best params
        """
        param_grid = {'C': np.logspace(-4, 4, 10)}
        clf = help.cv_param_estimation(train_df,
                                       LogisticRegression(max_iter=100000),
                                       param_grid,
                                       n_folds=3,
                                       feat_names=feat_names,
                                       type='LR')
        """
        Cross validation on train set
        """
        val_pr = help.validation(clf, train_df, feat_names)
        # Get equal error rate threshold
        GAMMA = help.get_gamma_ee(val_pr)

        """
        Train classifier
        """
        clf.fit(train_df[feat_names], train_df.TB_status.values)

        """
        Test classifier
        """
        TBI_probs, TBI_ref = help.test_model(clf, test_df, feat_names, GAMMA)

        for i in range(np.shape(TBI_probs)[0]):
            fpr, tpr, _ = roc_curve(TBI_ref, TBI_probs[i])
            mean_tpr[i] += interp(mean_fpr, fpr, tpr)
            mean_tpr[i][0] = 0.0

    mean_auc = []
    for i in range(2):
        # Normalize the mean_tpr
        m_tpr = mean_tpr[i]
        m_tpr /= N_FOLDS
        m_tpr[-1] = 1.0

        # Get AUC
        m_auc = auc(mean_fpr, m_tpr)
        mean_auc.append(m_auc)

    return mean_auc[0], mean_auc[1]





test_files = config.features_list

for test_csv in test_files:
    print ('\n\nRunning on ',os.path.basename(test_csv))
    df = pd.read_csv(test_csv, sep=';').dropna(thresh=2)
    df = df.drop(['ZCR','Kurtosis','LogE'], axis=1)
    # Get feature names (X)
    feat_names_all = list(df.columns.drop(["Study_Num", "TB_status", 'Bin_No', 'Cough_No']))
    if 'Win_No' in feat_names_all:
        feat_names_all.remove('Win_No')

    print ('\nGetting overall results...')
    # overall_ADS_AUC, overall_TIS_AUC = get_auc(df, test_csv, feat_names_all)
    overall_ADD_AUC_boot, overall_boot_se = get_bootstrap_auc(df, test_csv, feat_names_all, B=B)
    print ('done')
    
    
    ADS_AUC = []
    ADS_se = []
    segment_labels = []
    ## Madhu: I am making this section of codes for MFCC
    # if 'MFCC=' in test_csv:
    N_MFCC = int(test_csv.split('MFCC=')[-1].split('_')[0])
    for n in range(1, N_MFCC+1):
        feat_names = []
        for f in feat_names_all:
            if 'MFCC'+str(n) == f or 'MFCC_D'+str(n) == f or 'MFCC_2D'+str(n) == f:
                feat_names.append(f)
        print('\nEvaluating MFCC no: ', n, ' for selected feature: ', str(feat_names))
        AUC_boot, boot_se = get_bootstrap_auc(df, test_csv, feat_names, B=B)
        
        ADS_AUC.append(AUC_boot)
        ADS_se.append(boot_se)
    

    # # Number of filterbanks
    # N = len(feat_names_all)
    # # Hz increase per split
    # khz_inc = 22.1 / SPLITS
    # # filterbanks per split
    # b = N / SPLITS
    # ADS_AUC = []
    # ADS_se = []
    # segment_labels = []
    # for n in range(0, SPLITS):

    #     seg_label = "{:.1f}-{:.1f}".format(n*khz_inc, min(22.1,(n+1)*khz_inc))
    #     segment_labels.append(seg_label)

    #     print ('\nEvaluating: ', seg_label)

    #     start = int(n * b)
    #     end = int((n + 1) * b)

    #     feat_names = feat_names_all[start:end]

    #     # Get mean of these features
    #     # mean_auc = get_auc(df, test_csv, feat_names)
    #     AUC_boot, boot_se = get_bootstrap_auc(df, test_csv, feat_names, B=B)
        
    #     ADS_AUC.append(AUC_boot)
    #     ADS_se.append(boot_se)

    """
    Plot the results
    """
    sns.set(style='ticks')

    x = np.arange(0, N_MFCC, 1)
    width = x[1]-x[0]

    fig, ax = plt.subplots()

    rects = ax.bar(x, ADS_AUC, width, yerr=ADS_se,
                   error_kw=dict(ecolor='darkred', capsize=5, capthick=2))

    # for rect in rects:
    #     h = rect.get_height()
    #     ax.text(rect.get_x()+rect.get_with()/2., 1,05*h,
    #             '{}'.format(int(h)), ha='center', va='bottom')

    ax.axhline(y=overall_ADD_AUC_boot, color='darkorange')
    ax.axhline(y=overall_ADD_AUC_boot+overall_boot_se, color='darkorange',linestyle='--')
    ax.axhline(y=overall_ADD_AUC_boot-overall_boot_se, color='darkorange', linestyle='--')
    
    """
    Adjust the axis
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    ax.set_ylabel('ADS AUC')
    plt.xlabel('Frequency (kHz)')
    plt.xticks(x + width / 2, segment_labels, rotation=45)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.subplots_adjust(left=0.09, bottom=0.2, right=0.96)
    plt.show()
    
    fig_dir = '../data/LR_classifier/results/figures/feature_reduction/{}_splits/'.format(SPLITS)
    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)
    fig_name = fig_dir + 'MFCC={}_B={}.png'.format(N_MFCC, B)
    plt.savefig(fig_name)
    plt.close()




# import math 
# print(math.log(15))

# sr = 22050
# N_MFCC = 13

# freq_inc = (sr/2)/13

# freq_range = range(0, int(sr/2), int(freq_inc))
# mel_f = []
# for f in freq_range:
#     mel_f.append(1125*(math.log(1+(f/700))))




