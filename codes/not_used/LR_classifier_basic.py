#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 21:45:31 2020

@author: madhu
"""


import os
import sys
import numpy as np
import pandas as pd
from glob import glob
# from scipy.io.wavfile import read
import shutil
from natsort import natsorted
import pickle
import csv
import librosa
import sklearn
from operator import itemgetter
from scipy.stats import pearsonr
from scipy.io.wavfile import read, write 
from sklearn.linear_model import LogisticRegression
from scipy.stats import kurtosis
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, PredefinedSplit, GridSearchCV

import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from subprocess import Popen, PIPE
import random


import warnings

# Live on the wild side
pd.options.mode.chained_assignment = None
warnings.simplefilter(action = "ignore",category = FutureWarning)




N_MFCC = 13
N_lFBANK = 80


with open ('../data/patients_information', 'rb') as fp:
    patients_info = pickle.load(fp)

# data = pd.read_csv('../data/features.csv') 

data_MFCC = pd.read_csv('../data/features_MFCC='+str(N_MFCC)+'.csv') 
data_FBANK = pd.read_csv('../data/features_FBANK='+str(N_lFBANK)+'.csv') 

# data = pd.read_csv('../data/features.csv', index_col=0) 
data = data_MFCC
# data = data_FBANK



# """
# Using all data, but rather than computing the accuracy as 
# a measure of success, computing sensitivity, specificity and
# f score as measures of success.

# Then tweaking the threshold value gamma, used to perform prediction

# """

# def setup_names_and_labels(meta_data, px):

#     """
#     Function to get recording based labels
#     for all recordings in feat_csv

#     INPUT:
#     ======
#     meta_data:  numpy array containing all meta data
#                 with names in first column and labels
#                 in last column and one header line

#     px:         List with all recordings in audio dataset


#     OUTPUT:
#     =======



#     """

#     # list of all patient names in meta data
#     meta_px = fix_patient_names(meta_data[1:, 0])
#     # meta_y_px = map( float,list(meta_data[1:,-1]) )
#     # meta_y_px = list(meta_data[1:,-1])
#     meta_y_px = np.asarray(list(meta_data[1:,-1])).astype(float)
    
    

#     y_px = []
#     temp_px = []
#     for k in range(len(meta_px)):

#         for p in px:

#             if meta_px[k] in p:
#                 temp_px.append(meta_px[k])
#                 y_px.append(meta_y_px[k])

#             else:
#                 continue

#     px = temp_px

#     return px, y_px


def reserve_set(data, px):

    """

    Reserve 10% of the data for a different dataset

    Ensures that the reserved set contains examples from
    both classes. 

    Does not ensure the reserved set is balanced


    Inputs:
    -------
    data:   DataFrame with all data
    px:     List of patient names

    * px is the list of patients contained in data
    so if data has C_364_D and C_364_F then px will only have C_364


    Outputs:
    reserved_set:   Reserved dataset
    dev_data:       Remaining data

    """
    N = len(px)
    
    N_res = int(0.1 * N)
    
    
    dev_df = data.copy()
    res_df = pd.DataFrame(columns = list(data))

    res_px = []

    k = 0
    while k < N_res:
        
        # input("Press Enter to continue...")
        
        # Select a random patient
        rnd_idx = random.randint(0, N-1)
        
        # print('random int:', rnd_idx)
        # print('random patient: ', px[rnd_idx])
        # print('res_px is: ', res_px)
        # Dont take duplicates
        if (px[rnd_idx] not in res_px):
            
            neg_count = list(res_df["TB_Result"]).count(0)
            pos_count = list(res_df["TB_Result"]).count(1)

            # print ("neg_count:",neg_count)
            # print ("pos_count:",pos_count)
            
            """
            We need at least 3 samples from each class (0 and 1)
            to do cross validation with param estimation

            Thus after populating more than half the reserved set
            we check for this condition and loop until it's met.
            """
            if k >= int(N_res/2) and (pos_count < 3 or neg_count < 3) :
                # print('It is reducing k by 1')
                k -= 1

            else:
                # Get a temp dataframe of all data that matches the first recording
                # of the randomly selected patient
                temp_df = dev_df[dev_df["Study_Num"].str.contains(px[rnd_idx])]
                temp_idxs = temp_df.index
    
                # Save the temp dataframe
                res_df = res_df.append(temp_df)
    
                # Remove the save data from the dev dataset
                dev_df.drop(temp_idxs,inplace = True)
    
                # Save the recording 
                res_px.append(px[rnd_idx])
                # print('random patient added: ', px[rnd_idx])
                k += 1

    print ("num classes in reserved set:",len(np.unique(res_df["TB_Result"])))

    dev_px = [p for p in px if p not in res_px]

    return res_df, res_px, dev_df, dev_px




def LOOV(data, px, y_px):
    
    global GAMMA
    
    """
    Leave One Out Validation
    
    ** In fact, this function performs Leave N-Out Validation, 
    leaving out N amount of patients from the training set in order
    to split the dev set into N_FOLDS folds. 
    
    INPUTS:
    =======
    data:       DataFrame containing all data (sorted according to StudyNum)
    px:         List of patients contained in data
    y_px:       Labels of px - used for stratified k fold splitting of px
    
    
    OUTPUTS:
    ========
    
    acc:        Accuracy of the Logistic Regression model on data
    sens:       Sensitivity " .. "
    spec:       Specificity " .. "

    """

    # Reserve Cross Validation set from dev set for param estimation
    cv_data, cv_px, train_data, train_px = reserve_set(data, px)

    # print "Cross Validation data:"
    # print cv_data[["StudyNum","TBResult"]]
    # Get a Logistic Regression model with optimized params
    print ("Getting optimal LR model...")
    LRM = param_estimation(cv_data)

    pred = []               # list of prediction (0 or 1)
    probs = []              # list of probabilities (P(Y=1|X))
    y_ = []                 # labels in same order as pred
    test_rec_list = []      # list of recordings being tested

    # skf = StratifiedKFold(np.zeros(len(y_px)), y_px, n_splits = N_FOLDS, shuffle = True)
    # skf = StratifiedKFold(np.asarray(y_px), n_splits = N_FOLDS, shuffle = True)
    skf = StratifiedKFold(n_splits = N_FOLDS, shuffle = True)
    
    # print ("Running Stratified K-Fold splits")
    for train_idx, test_idx in skf.split(np.zeros(len(y_px)), y_px):
        # print("TRAIN:", train_idx, "TEST:", test_idx)
        
        # Split data into training and testing set
        X_train, y_train, X_test, y_test, test_recs = leave_out_folds(data, px, train_idx, test_idx)

        n,d = X_test.shape

        # if (n == 0 or d == 0):

        #     print "ALL_IDX:"
        #     print list(data.index)
        #     print "X_train:"
        #     print "indexes:",train_idx
        #     print X_train.shape
        #     print "X_test"
        #     print "indexes:",test_idx
        #     print X_test.shape

        # ======= TRAIN LRM =======
        LRM.fit(X_train, y_train)

        ## Save the probabilities, predictions and correct order labels
        # Get the probabilities of X_test
        probs.append(list(LRM.predict_proba(X_test)[:,1]))
        # Get the predictions of X_test
        pred.append(list(LRM.predict(X_test)))
        y_.append(list(y_test))

        # Also save the list of recording names in test set
        test_rec_list.append(test_recs)


    probs = [item for sublist in probs for item in sublist]
    pred = [item for sublist in pred for item in sublist]
    y_ = [item for sublist in y_ for item in sublist]
    test_rec_list = [item for sublist in test_rec_list for item in sublist]

    # Get ROC curve values
    # Madhu: I don't think we need to specify pos_label as the binary labels are : {0, 1}
    fpr, tpr, thresholds = roc_curve(np.asarray(y_).astype(int), np.asarray(probs))
    # fpr, tpr, thresholds = roc_curve(np.asarray(y_).astype(int), np.asarray(probs), pos_label=0)
    
    # area = roc_auc_score(np.asarray(y_).astype(int), np.asarray(probs))
    

    """
    Determine threshold to minimise the difference
    between tpr and (1-fpr)
    * (1-fpr) is tnr which is specificity 
    """
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),
                        'tpr' : pd.Series(tpr, index = i),
                        '1-fpr' : pd.Series(1-fpr, index = i),
                        'tf' : pd.Series(tpr - (1-fpr), index = i),
                        'thresholds' : pd.Series(thresholds, index = i)
                        })
    idx = (roc.tf).abs().argmin()
    thresh = roc.thresholds.iloc[idx]
    
    
    """
    Perform classification with new threshold
    """
    pred = []
    for p in probs:
        if p >= thresh:
            pred.append(1)
        else:
            pred.append(0)


    # Lastly, update the threshold value for predictions i.e. set GAMMA as the new threshold value 
    GAMMA = thresh

    """ 
    Get results using optimized threshold
    """
    # acc = eval_model(pred, y_)
    acc,sens,spec = eval_model(pred, y_, probs = probs, test_recs = test_rec_list, TBI = 1, return_ = 1)
    
    return LRM, [acc,sens,spec]

def param_estimation(cv_data):


    y_cv = cv_data["TB_Result"]

    drop = ["Study_Num","TB_Result"]

    X_cv = cv_data.drop(drop, axis = 1)

    # LR_model = LogisticRegression()
    LR_model = LogisticRegression(max_iter=1000000)

    LR_param_grid = {'C': np.logspace(-3,3,7)}

    LR_params = run_gridsearch(X_cv, y_cv.astype('int'), LR_model, LR_param_grid, cv_fold = 2)

    LR_model.set_params(**LR_params)

    return LR_model


def run_gridsearch(X,y,model,params,cv_fold):    
    """
    Grid search to find the optimal parameters
    """

    grid_search = GridSearchCV(model,param_grid = params,cv = cv_fold)
    # grid_search = GridSearchCV(model,param_grid = params)
    
    # print('X has values:', X)
    # print('y has values:', y)
    
    grid_search.fit(X,y)

    # gs_opt_params = report(grid_search.grid_scores_,n_top=1,print_=0)
    # gs_opt_params = report(grid_search.cv_results_,n_top=1,print_=0)
    
    gs_opt_params = grid_search.best_params_

    return gs_opt_params


def report(grid_scores, n_top=3,print_ = 0):
    """Report top n_top parameters settings, default n_top=3.

    Args
    ----
    grid_scores -- output from grid or random search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """
    top_scores = sorted(grid_scores,key=itemgetter(1),reverse=True)[:n_top]

    return top_scores[0].parameters    

def leave_out_folds(X, px, train_idx, test_idx):
    """
    This function creates a training and testing dataset
    from the dataset X, by leaving out all the data from
    one patient. 

    Inputs:
    =================
    X:          Complete dataset
    px:         List of patient names in dev set
    train_idx:  Indexes for px of patients to be included in train set
    test_idx:   Indexes for px of patients to be included in test set

    Returns:
    =================
    X_train:    Training data
    y_train:    Training labels
    ---------
    X_test:     Testing data
    y_test:     Testing labels
    ---------
    test_names: List of recordings in test set
    =================



    UPDATE: Changing to leave out k folds
    """

    # Generate training set:
    X_train = [ X[ X["Study_Num"].str.contains(px[k]) ] for k in train_idx]
    X_train = pd.DataFrame(np.vstack(X_train),columns = list(X))

    # Generate testing set
    X_test = [ X[X["Study_Num"].str.contains(px[k])] for k in test_idx]
    X_test = pd.DataFrame(np.vstack(X_test),columns = list(X))


    if (X_test.shape[0] == 0):
        print ("Test idx",test_idx)
        for k in test_idx:
            print (k,px[k],X[X["Study_Num"].str.contains(px[k])])
        exit()


    # Always sort the dataframes according to StudyNum
    X_test.sort_values(by = ['Study_Num'], inplace = True)
    X_train.sort_values(by = ['Study_Num'], inplace = True)
    
    # X_test.sort(columns = "StudyNum",inplace = True)
    # X_train.sort(columns = "StudyNum",inplace = True)

    # Get a list of all the recording names in the test set (sorted)
    test_names = list(X_test["Study_Num"])

    # Get labels of train and test sets
    
    y_test = np.asarray(list(X_test["TB_Result"])).astype(float)    
    y_train = np.asarray(list(X_train["TB_Result"])).astype(float)
    # y_test = map(float,list(X_test["TBResult"]))
    # y_train = map(float,list(X_train["TBResult"]))

    # Remove these two columns from the dataframe
    drop_ = ["TB_Result","Study_Num"]
    X_test.drop(drop_,axis = 1,inplace = True)
    X_train.drop(drop_,axis = 1,inplace = True)

    # Convert the dataframes to float
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    # X_train.convert_objects(convert_numeric = True)
    # X_test.convert_objects(convert_numeric = True)

    return X_train, y_train, X_test, y_test, test_names


def train_model(data, LRM = None):

    # y = map(float, list(data["TBResult"]))
    y = np.asarray(list(data["TB_Result"])).astype(float)    
    X = data.drop(["Study_Num","TB_Result"],axis = 1)

    if LRM == None:
        LRM = LogisticRegression(C = C_REG)

    LRM.fit(X,y)

    return LRM

def test_model(LRM, test_data, return_ = 0, TBI = 0, save = 0):

    """
    Evaluate a trained Logistic Regression model


    Inputs:
    =======
    LRM:        Trained Logistic Regression Model
    test_data:  Data to test the LRM on
    return:     Flag - To return [spec, sens, acc] or just acc
    TBI:        Flag - To compute results using TBI or not

    """

    # Use the same threshold that was
    # selected when optimizing on the training set
    global GAMMA    

    # Get the labels
    # y = map(float,list(test_data["TBResult"]))
    y = np.asarray(list(test_data["TB_Result"])).astype(float)
    

    # Get the names of the recordings in the test set
    test_recs = list(test_data["Study_Num"])

    # Keep the feature data for training
    X = test_data.drop(["Study_Num","TB_Result"], axis = 1)

    probs = LRM.predict_proba(X)[:,1]

    # perform prediction with optimizad threshold
    pred = []
    for p in probs:
        if p >= GAMMA:
            pred.append(1)
        else:
            pred.append(0)

    if TBI == 0:
        acc,sens,spec = eval_model(pred, y, return_ = 1)

    else:
        acc,sens,spec = eval_model(pred, y, probs = probs, test_recs = test_recs, TBI = 1, return_ = 1, save = save)


    if return_ == 0:
        return acc
    else:
        return [acc, sens, spec]


def eval_model(pred, y_test, probs = [], test_recs = [], TBI = 0, return_ = 0, save = 0):

    global GAMMA
    """
    This function determines the relationship between pred and y_test
    as a measure of how well predictions have been made by a classifier


    INPUTS:
    =======
    pred:       List of predictions (binary)
    y_test:     List of correct labels (binary)
    probs:      List of estimated probabilities
    test_rec:   List of recordings that predictions have been made for
    TBI:        Flag - to compute and return TB Index values or not
    return_:    Flag - to return sensitivity, specificity and acc or just acc
    save:       Flag - to save probabilities (on a recording level) or not [Only applies when TBI = 1]
    
    *Note test_rec is not sorted, but corresponds to the order of y_test and pred

    OUTPUTS:
    ========

    if TBI == 0:    Compute results on cough level

    if TBI != 0:    Compute all results on a recording level
                    In other words, award 1 if more then 50%
                    of the predictions in a recording have been
                    correct else return 0.


    if return_ == 0:
    ----------------

    acc:        Accuracy

    If return_ != 0:
    ----------------

    sens:       Sensitivity
    spec:       Specificity
    acc:        Accuracy

    """

    # first check that the same number of predictions
    # are made than we have test labels for
    n_pred = len(pred)
    n_test = len(y_test)

    if(n_pred != n_test):
        print ("Predicted labels and test labels dont have the same dimensions!")
        print ("Predicted: ", n_pred, "; Tests: ", n_test)
        sys.exit()


    if TBI == 0:
        N = n_pred

        tp = 0.
        tn = 0.
        fp = 0.
        fn = 0.

        for k in range(0, N):

            p = pred[k]
            l = y_test[k]

            if(p == 1 and l == 1):
                tp += 1
            elif (p == 1 and l == 0):
                fp += 1
            elif (p == 0 and l == 0):
                tn += 1
            elif (p == 0 and l == 1):
                fn += 1

        if tp == 0 or tn == 0:

            if tp == 0 and tn != 0:
                sens = 0
                spec = tn / (tn + fp)

            if tn == 0 and tp != 0:
                spec = 0
                sens = tp / (tp + fn)

        else:
            sens = tp / (tp + fn)
            spec = tn / (tn + fp)


        acc = (tp+tn) / N

        # if return_ == 0:
        #     return acc
        # else:
        #     return acc, sens, spec

    elif TBI == 1:

        "Calculate results on a recording level"

        # y_test = map(int, y_test)
        y_test = np.asarray(y_test).astype(int)
        
        # perform a check first
        if (len(test_recs) != n_pred):
            print ("Recordings and predictions / labels dont match:")
            print ("Num recordings:",len(test_recs)," Num predictions,labels:",len(pred),len(y_test))
            exit()


        i = np.arange(len(test_recs))
        df = pd.DataFrame({"Recording": pd.Series(test_recs,index = i),
                            "Prediction": pd.Series(pred,index = i),
                            "Reference": pd.Series(y_test,index = i),
                            "Probabilities": pd.Series(probs,index = i)
                            })
        df = df.sort_values(by=['Recording'])
        
        # df = pd.DataFrame({"Recording": pd.Series(test_recs,index = i),
        #                     "Prediction": pd.Series(pred,index = i),
        #                     "Reference": pd.Series(y_test,index = i),
        #                     "Probabilities": pd.Series(probs,index = i)
        #                     }).sort(columns = "Recording")
        
        """
        I changed the TBI method:
        
        Take the average of all the probabilities of coughs being TB
        (P(Y=1|X)) over all coughs in a recording.
        If the avg probability is >= GAMMA then make the diagnosis
        as TB -> ie the patient probably has TB
        Else make the diagnosis as Not TB. 
        
        
        Further to do: 
        
        Divide the diagnosis into bins: high, mid, low prob. 
        
        """
        
        rec_list = []
        y_list = []
        TBI_list = []
        for name, group in df.groupby("Recording"):

            # pred = list(group["Prediction"])
            # ref = list(group["Reference"])

            rec_list.append(name)

            # The label of this recording
            l = group["Reference"].iloc[0]
            y_list.append(l)

            
            prob = sum( list(group["Probabilities"])) / float(len(group["Probabilities"]) )

            TBI_list.append(prob)

        """
        TBI_list contains the average probability P(Y=1) over all coughs in a recording
            - Thus contains one probability, the probability that this recording was from
              a TB+ patient.

        TBI_list is sorted in terms of recording name
        """



        if (save == 1):
            """
            Save the probabilities of each recording being positive (TBI_list) to disk
            """
            # # If the file exists
            # if (os.path.isfile(f_out)):

            #     # check if any of the current recordings have already
            #     # been tested
            #     tmp_test_data = pd.read_csv(f_out,sep=";")

            #     tmp_test_recs = tmp_test_data["StudyNum"]

            #     for rec in rec_list:
            #         if rec in tmp_test_recs:
            #             print rec
            #             exit()

            # For each recording
            for i in range(len(TBI_list)):
                # name of recording
                p = rec_list[i]
                # label of recording (actual diagnosis of patient)
                l = y_list[i]
                # probability this recording is from a TB+ patient
                tbi = TBI_list[i]

                line = str(p)+";"+str(l)+";"+str(tbi)+"\n"



        """
        Evaluate the model using the TBI probs
        """

        N = len(TBI_list)

        tp = 0.
        tn = 0.
        fp = 0.
        fn = 0.

        diagnosis_list = []
        # For each recording
        for k in range(N):

            # if prob is >= threshold
            if TBI_list[k] >= GAMMA:
                diagnosis = 1
            else:
                diagnosis = 0

            diagnosis_list.append(diagnosis)

            # Actual diagnosis
            l = y_list[k]

            if(diagnosis == 1 and l == 1):
                tp += 1
            elif (diagnosis == 1 and l == 0):
                fp += 1
            elif (diagnosis == 0 and l == 0):
                tn += 1
            elif (diagnosis == 0 and l == 1):
                fn += 1


        # for i in range(len(TBI_list)):
        #     print "Recording:",rec_list[i],"TBI:",TBI_list[i],"\tDiagnosis:",diagnosis_list[i]

        sens = 0
        spec = 0

        if tp == 0 or tn == 0:

            if tp == 0 and tn != 0:
                sens = 0
                spec = tn / (tn + fp)

            if tn == 0 and tp != 0:
                spec = 0
                sens = tp / (tp + fn)

        else:
            sens = tp / (tp + fn)
            spec = tn / (tn + fp)


        acc = (tp+tn) / N

        # if (sens == 0 or spec == 0):
        #     print "\n\n"
        #     print "GAMMA = ",GAMMA
        #     print "Sensitivity:",sens
        #     print "Specificity",spec
        #     print "TP:",tp
        #     print "TN",tn
        #     print "FP:",fp
        #     print "FN",fn
        #     print len(TBI_list)
        #     for i in range(len(TBI_list)):
        #         print "Recording:",rec_list[i],"TBI:",TBI_list[i],"\tDiagnosis:",diagnosis_list[i]
        #     exit()

        # if return_ == 0:
        #     return acc
        # else:
        #     return acc, sens, spec

    else:
        print ("Please select '0' or '1' for TBI")
        exit()
    
    if return_ == 0:
        to_return =  acc
    else:
        print('acc :', acc)
        print('sens :', sens)
        print('spec :', spec)
        to_return = acc, sens, spec

    return to_return


def get_data_skewness(y, print_ = 0):
    """
    Data is skew (more pos than neg) and I want to write a function
    to plot the skewness percentage versus the classification accuracy.

    INPUT:
    ======

    y:      List containing labels
    print_  Flag - 1: print skewness to console
                 - 0: dont print, just return


    OUTPUT:
    =======

    skew:   Skewness of the dataset

    """

    N = len(y)

    count_pos = 0
    count_neg = 0
    count_other = 0

    for k in range(0, N):

        if(y[k] == 1):
            count_pos += 1

        elif(y[k] == 0):
            count_neg += 1

        else:
            count_other += 1

    skew = (float(count_pos) / float(N)) * 100

    if print_ != 0:
        print (skew,"% positive samples")

    return skew


def exit():
    sys.exit()





    """
    This is only a Logistic Regression classifier

    Process:

    For N times:

        - split dataset into test and dev sets
            - test set is randomly selected
                - Data is split on patient level (ie all recordings
                    per random patient is included)

        - compute p values of all features in complete dataset
        - remove all features from test and dev set is p < 0.05
        - Train and validate on dev set using stratified k fold
            - Same test/train splitting mechanism is used as before
            - compute optimal threshold value GAMMA using ROC analysis

        - Train model on all data in dev set

        - Test trained model using independant test set
            - Perform prediction from probabilities using GAMMA
              computed in LOOV

        - Because GAMMA minimizes difference between spec and sens
          the accuracy (tp + tn)/N will be ~= spec and sens
        - Save accuracy on test set

    Take mean of all test accuracy scores.



    UPDATE: CHANGING TO MAKE A PATIENT LEVEL DIAGNOSIS

    In this version, a diagnosis is made by taking the average probability 
    of TB for all coughs in a recording, then using the updated GAMMA making
    a diagnosis of TB if the avg prob >= GAMMA.

    """









# Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
C_REG = 0.001
GAMMA = 0.5
N_FOLDS = 10
N_RUNS = 50


#  column names
feat_names = list(data)[3:data.shape[1]]

# sort the data according to recording name
# data = data.sort(columns = "StudyNum")

# Calculate the skewness of the audio dataset
# skew = get_data_skewness( map( float,list(data["TBResult"])), print_ = 1)
skew = get_data_skewness( np.asarray(list(data["TB_Result"])).astype(float), print_ = 1)

# # Read in meta data (just to get labels)
# meta_data = np.loadtxt(Meta_csv, delimiter=";", dtype=str)

# # Get recording based labels (px and y_px correspond in sort order)
# px, y_px = setup_names_and_labels(meta_data, np.unique(list(data["StudyNum"])))

# px: patients_list
# y_px: TB_results_list


LR_models = []
overall_LOO_acc = []
overall_test_results = []
overall_TBI_test_results = []

test_px_list = []


patients_list = []
TB_results_list = []
for pat in patients_info:
    # print(pat)
    patients_list.append(pat)
    TB_results_list.append(patients_info[pat]['TB_status'])


for i in tqdm(range(N_RUNS)):

    """
    Reserve a random subset of the data for testing.

    Then calculate p values for each feature in the
    remaining dataset (Train set)

    Drop all features with p values lower than 0.05

    Train and validate on the train set with reduced
    feature set and calculate threshold value GAMMA to
    balance specificity and sensitivity. Save that GAMMA
    as global variable.


    Then test on independant test set using optimized
    threshold GAMMA and the same regularization rate
    used in training.

    Do this 10 times and save accuracy of each iteration

    """
    
    # reserve 10% of the data as an independent test set
    # test_data, test_patList, dev_data, dev_patList = reserve_set(data, patients_list)
    test_data, test_patList, dev_data, dev_patList = reserve_set(data, sorted(patients_list)[17:len(patients_list)])
    
    print('Test-patients are: ', test_patList)
    
    test_px_list.append(test_patList)

    # get test and dev set recording-based labels for stratification
    y_test_patList = [TB_results_list[patients_list.index(p)] for p in test_patList]
    y_dev_patList = [TB_results_list[patients_list.index(p)] for p in dev_patList]
    
    
    """
    Feature Selection
    """
    # Get p values of all features in dev_data
    # p_vals = [pearsonr(dev_data[col].astype(float), map(float,list(dev_data["TBResult"]) ) )[1] for col in feat_names[1:-1]]
    p_vals = [pearsonr(dev_data[col].astype(float), np.asarray(list(dev_data["TB_Result"])).astype(float))[1] for col in feat_names[0:len(feat_names)]]
    
    # # Investigate p-values
    # for col in feat_names:
    #     print(col)
    #     print(pearsonr(dev_data[col].astype(float), np.asarray(list(dev_data["TB_Result"])).astype(float))[1])
    
    # Get the features with a p value lower than 0.05
    new_feat_names = [feat_names[k] for k in range(len(p_vals)) if p_vals[k] < 0.05]
    
    # Keep only that data
    new_data = dev_data[new_feat_names]
    
    # but add the patient names and Labels
    new_data['Study_Num'] = dev_data["Study_Num"]
    new_data["TB_Result"] = dev_data["TB_Result"]
    
    # update the test dataset too
    new_test_data = test_data[new_feat_names]
    new_test_data["Study_Num"] = test_data["Study_Num"]
    new_test_data["TB_Result"] = test_data["TB_Result"]
    
    
    """ 
    Leave One Out (actually leave k out) Validation
    on Dev set
    """
    LR_model, loo_acc = LOOV(new_data, dev_patList, y_dev_patList)
    
    LR_models.append(LR_model)
    
    """
    Train Logistic Regression model on entire dev set

    """
    
    LRM = train_model(new_data, LR_model)

    
    # Test on held out set without using TBI Method
    test_results = test_model(LRM, new_test_data, return_ = 1)

    # Test on held out set using TBI Method
    TBI_test_results = test_model(LRM, new_test_data, return_ = 1, TBI = 1, save = 1)

    # Save the accuracies
    overall_test_results.append(test_results)
    overall_TBI_test_results.append(TBI_test_results)
    overall_LOO_acc.append(loo_acc)
    

# Get the average results over all runs
overall_LOO_acc = np.vstack(overall_LOO_acc)
means =  np.mean(overall_LOO_acc,axis = 0)
acc = means[0]
sens = means[1]
spec = means[2]

print ("==========================")
print ("N_FOLDS = ",N_FOLDS)
print ("N_RUNS = ",N_RUNS)

print ("\n\nOVERALL RESULTS:")
print ("\nLOOV Results:")
print ("\t\t\t Sensitivity:",spec)
print ("\t\t\t Specificity:",sens)
print ("\t\t\t Accuracy:",acc,"\n\n")

overall_test_results = np.vstack(overall_test_results)
means =  np.mean(overall_test_results,axis = 0)
acc = means[0]
sens = means[1]
spec = means[2]

print ("\nIndependent Test Sets Results:")
print ("\t\t\t Sensitivity:",spec)
print ("\t\t\t Specificity:",sens)
print ("\t\t\t Accuracy:",acc,"\n\n")

overall_TBI_test_results = np.vstack(overall_TBI_test_results)
means =  np.mean(overall_TBI_test_results,axis = 0)
acc = means[0]
sens = means[1]
spec = means[2]

print ("\nIndependent TBI Based Test Sets Results:")
print ("\t\t\t Sensitivity:",spec)
print ("\t\t\t Specificity:",sens)
print ("\t\t\t Accuracy:",acc,"\n\n")

print ("==========================")

# Get a list of patients that haven't been independantly tested
test_px_list = [item for sublist in test_px_list for item in sublist]
not_tested = [p for p in patients_list if p not in list(set(test_px_list))]
print ("List of patients not tested:",not_tested)








