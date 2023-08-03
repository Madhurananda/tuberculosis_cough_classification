#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:05:57 2020

@author: madhu
"""

import os
import sys
import numpy as np
import pandas as pd
from glob import glob
# from scipy.io.wavfile import read
import shutil
import pickle
import csv
import librosa
import sklearn

from scipy.stats import pearsonr
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from sklearn.linear_model import LogisticRegression
from scipy.stats import kurtosis
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.signal import hamming

import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from subprocess import Popen, PIPE
import random

sys.path.insert(0, './Renier_scripts')
# from log_filterbanks import log_filterbanks as log_fbanks
from log_filterbanks import log_filterbanks
from helper import *
from feature_extraction_MFCC import calc_mfcc, get_mfcc_means, calc_MFCC_D_A

import warnings

# # Live on the wild side
# pd.options.mode.chained_assignment = None
# warnings.simplefilter(action = "ignore",category = FutureWarning)


def cepstral_mean_subtraction(mfccs,means):
	
	mfccs_ = []

	k = 0
	for coeff in mfccs:
		coeff_ = [x - means[k] for x in coeff]
		mfccs_.append(coeff_)
		k += 1

	return np.array(mfccs_)


def zero_crossing_rate_BruteForce(wavedata):
    
    zero_crossings = 0
    
    number_of_samples = len(wavedata)
    for i in range(1, number_of_samples):
        
        if ( wavedata[i - 1] <  0 and wavedata[i] >  0 ) or \
           ( wavedata[i - 1] >  0 and wavedata[i] <  0 ) or \
           ( wavedata[i - 1] != 0 and wavedata[i] == 0):
                
                zero_crossings += 1
                
    zero_crossing_rate = zero_crossings / float(number_of_samples - 1)

    return zero_crossing_rate

def get_LogEnergy(audio):

	eps = 0.0001
	N = len(audio)
	energy = abs(audio**2)
	LogE =10 * np.log10(eps + sum(energy) / float(N))

	return LogE



## Step 1: 

data_path = "../data/raw_data/sorted_recording/"

path_contents = os.listdir(data_path)


for i in path_contents:
    if i.endswith('.xlsx'):
        data_description = os.path.join(data_path, i)


# Read the XLSX file as panda dataframe 
df = pd.read_excel (data_description)
print (df)

patients = []
TB_status = []
for pat in df['Subject ID']:
    print(pat)
    patients.append(pat)

for tb_st in df['Final_TB_Result (1 = TB, 0 = No-TB)']:
    TB_status.append(tb_st)



## One patinet had his recording in two seperate take
pat_take_1 = os.path.join(data_path, 'Wu0392_1/')
pat_take_2 = os.path.join(data_path, 'Wu0392_2/')

output = os.path.join(data_path, 'Wu0392/')

wav_files_1 = os.listdir(pat_take_1)

wav_files_2 = os.listdir(pat_take_2)

# length of wav_files_1 and wav_files_2 will be the same 
for i in range(0, len(wav_files_2)):
    
    if wav_files_1[i].endswith('_Tr1.wav'):
        new_wav_1_tr_1 = wav_files_1[i]
    if wav_files_2[i].endswith('_Tr1.wav'):
        new_wav_2_tr_1 = wav_files_2[i]
    
    if wav_files_1[i].endswith('_Tr2.wav'):
        new_wav_1_tr_2 = wav_files_1[i]
    if wav_files_2[i].endswith('_Tr2.wav'):
        new_wav_2_tr_2 = wav_files_2[i]
    
    if wav_files_1[i].endswith('_Tr3.wav'):
        new_wav_1_tr_3 = wav_files_1[i]
    if wav_files_2[i].endswith('_Tr3.wav'):
        new_wav_2_tr_3 = wav_files_2[i]


y_1, sr = librosa.load(pat_take_1+new_wav_1_tr_1, sr=48000)
y_2, sr = librosa.load(pat_take_2+new_wav_2_tr_1, sr=48000)
new_wav_tr_1 = list(y_1) + list(y_2)

sf.write(output+'Wu0392_Tr1.wav', new_wav_tr_1, sr)


y_1, sr = librosa.load(pat_take_1+new_wav_1_tr_2, sr=48000)
y_2, sr = librosa.load(pat_take_2+new_wav_2_tr_2, sr=48000)
new_wav_tr_2 = list(y_1) + list(y_2)

sf.write(output+'Wu0392_Tr2.wav', new_wav_tr_2, sr)


y_1, sr = librosa.load(pat_take_1+new_wav_1_tr_3, sr=48000)
y_2, sr = librosa.load(pat_take_2+new_wav_2_tr_3, sr=48000)
new_wav_tr_3 = list(y_1) + list(y_2)

sf.write(output+'Wu0392_Tr3.wav', new_wav_tr_3, sr)


##############################################################################
### Extract features #####


patients_info = {}

sorted_patient = []
for dir in path_contents:
    print(dir)
    if os.path.isdir(data_path + dir):
        p = dir
        print('Patient: ', p)
        sorted_patient.append(p)
        
        
        # if p == 'Wu0392' or p == 'Wu0395' or p == 'Wu0412':
        if p == 'Wu0392':
            continue
        
        patients_info[p] = {}
        # Get coughs from annotation values
        files_in_dir = os.listdir(data_path+p)
        for fl in files_in_dir:
            if fl.endswith('.eaf'):
                annot_file = fl
            elif fl.endswith('_Tr1.wav'):
                wav_Tr1_file = fl
            elif fl.endswith('_Tr2.wav'):
                wav_Tr2_file = fl
            elif fl.endswith('_Tr3.wav'):
                wav_Tr3_file = fl
        
        f = open(data_path+p+'/'+annot_file, "r")
        annot_file_content = f.read()
        
        list_info = annot_file_content.split('TIME_SLOT_ID=')
        
        time_labels = []
        time_stamps = []
        
        for ind in range(1, len(list_info)):
            
            if ind == (len(list_info) - 1):
                time_labels.append( list_info[len(annot_file_content.split('TIME_SLOT_ID='))-1].split('</TIME_ORDER>')[0].split('TIME_VALUE=')[0].split('"')[1] )
                time_stamps.append( list_info[len(annot_file_content.split('TIME_SLOT_ID='))-1].split('</TIME_ORDER>')[0].split('TIME_VALUE=')[1].split('"/>\n')[0].split('"')[1] )
            else:
                time_labels.append( list_info[ind].split('" TIME_VALUE="')[0].split('"')[1] )
            
                time_stamps.append( list_info[ind].split('" TIME_VALUE="')[1].split('"/>\n        <TIME_SLOT')[0] )
        
        # Check if the lengths of the time_labels and time_stamps
        if len(time_labels) != len(time_stamps):
            print('\n\nThere is something very wrong with cough labels extraction.')
            break
        
        if len(time_labels)%2 != 0:
            print('\n\nThe coughs do not have a even number of labels. Please check ... ')
            break
        
        # create coughs and save its information in the dictionary
        for c in range(0, int(len(time_labels)/2)):
            # print(c+1)
            patients_info[p]['cough_'+str(c)] = {}
            patients_info[p]['cough_'+str(c)]['start_time'] = time_stamps[2*c]
            patients_info[p]['cough_'+str(c)]['end_time'] = time_stamps[(2*c+1)]
            
            y_mc, sr = librosa.load(data_path+p+'/'+wav_Tr1_file, offset=int(time_stamps[2*c])/1000, duration=(int(time_stamps[(2*c+1)]) - int(time_stamps[2*c]))/1000, sr=48000)
            y_st_1, sr = librosa.load(data_path+p+'/'+wav_Tr2_file, offset=int(time_stamps[2*c])/1000, duration=(int(time_stamps[(2*c+1)]) - int(time_stamps[2*c]))/1000, sr=48000)
            y_st_2, sr = librosa.load(data_path+p+'/'+wav_Tr3_file, offset=int(time_stamps[2*c])/1000, duration=(int(time_stamps[(2*c+1)]) - int(time_stamps[2*c]))/1000, sr=48000)
            
            patients_info[p]['cough_'+str(c)]['audio_mc'] = y_mc
            patients_info[p]['cough_'+str(c)]['audio_st1'] = y_st_1
            patients_info[p]['cough_'+str(c)]['audio_st2'] = y_st_2
            patients_info[p]['cough_'+str(c)]['sr_rate'] = sr
            
            # Also, write the coughs into the subfolder: 'coughs'
            if os.path.isdir(data_path + p+'/coughs_mc/'):
                print('Main mic cough dir exists')
            else:
                print('Creating main mic cough directory: ', data_path + p+'/coughs_mc/')
                os.makedirs(data_path + p+'/coughs_mc/')
            sf.write(data_path + p+'/coughs_mc/cough_'+str(c)+'.wav', y_mc, sr)
            
            if os.path.isdir(data_path + p+'/coughs_st-1/'):
                print('Stethoscope (channel 1) cough dir exists')
            else:
                print('Creating stethoscope (channel 1) cough directory: ', data_path + p+'/coughs_st-1/')
                os.makedirs(data_path + p+'/coughs_st-1/')
            sf.write(data_path + p+'/coughs_st-1/cough_'+str(c)+'.wav', y_st_1, sr)
            
            if os.path.isdir(data_path + p+'/coughs_st-2/'):
                print('Stethoscope (channel 2) cough dir exists')
            else:
                print('Creating Stethoscope (channel 2) cough directory: ', data_path + p+'/coughs_st-2/')
                os.makedirs(data_path + p+'/coughs_st-2/')
            sf.write(data_path + p+'/coughs_st-2/cough_'+str(c)+'.wav', y_st_2, sr)
            
            


## I need to sort out the patient 'Wu0392' as his recordings were taken in two shots
len(patients_info['Wu0392_1'])
len(patients_info['Wu0392_2'])

temp_coughs_1 = patients_info['Wu0392_1']
# temp_coughs_2 = patients_info['Wu0392_2']

# temp_coughs_1.update(temp_coughs_2)


count_start = 43 # as first take has 43 coughs in it. 

temp_coughs = {}
for i in range(0, len(patients_info['Wu0392_2'])):
    temp_coughs['cough_'+str(count_start)] = patients_info['Wu0392_2']['cough_'+str(count_start-43)]
    count_start +=1


temp_coughs_1.update(temp_coughs)

patients_info['Wu0392'] = temp_coughs_1

del patients_info['Wu0392_1']
del patients_info['Wu0392_2']

for pat in patients_info:
    print ("Getting Cepstral mean for: ",pat)
    
    for p in patients:
        if pat == p:
            patients_info[pat]['TB_status'] = TB_status[patients.index(p)]
            patients.index(p)
    


updated_patient_info = {}

for pat in patients_info:
    # print(pat)
    if patients_info[pat]['TB_status'] == 1:
        new_pat_name = 'TB_' + pat
    elif patients_info[pat]['TB_status'] == 0:
        new_pat_name = 'NTB_' + pat
    
    updated_patient_info[new_pat_name] = patients_info[pat]

del patients_info

# Save the patients data
with open('../data/raw_data/patients_information', 'wb') as fp:
    pickle.dump(updated_patient_info, fp, protocol=4)

with open ('../data/raw_data/patients_information', 'rb') as fp:
    patients_info = pickle.load(fp)


# TB_count = 0
# NTB_count = 0
# for key in patients_info:
#     if key.startswith('TB'):
#         TB_count += 1
#     elif key.startswith('NTB'):
#         NTB_count += 1


################# Step 2: Extract MFCC and log Filterbank features: ###########
"""
This script extract the following features from a list of coughs:

1. MFCCs
2. Zero Crossing Rate (ZCR)
3. Kurtosis
4. Log Energy
5. Log Filterbank

This has been modified so that it can extract features just like Renier did in his thesis

"""

# N = 2048
# M = 2048
# B = 1
# N_FBANKS = 140

N_MFCCs = range(13, (13*3)+1, 13)
# N_MFCCs = [39]

N_FBANKS = list(range(40, 210, 20))
N_frames = [2**9, 2**10, 2**11, 2**12] # This should be equal to M and N
Bs = range(1, 5, 1)
Avgs = range(2)



for N_MFCC in tqdm(N_MFCCs):
    
    # features_vectors_MFCC[pat][str(N_MFCC)] = {}
    
    for N_frame in N_frames:
        
        # features_vectors_MFCC[pat][str(N_MFCC)][str(N_frame)] = {}
        
        for B in Bs:
        
            # features_vectors_MFCC[pat][str(N_MFCC)][str(N_frame)][str(B)] = {}
            
            for Avg in Avgs:
                
                feat_colNames = ["Study_Num",'Bin_No','Cough_No','Win_No',"ZCR","Kurtosis","LogE"]
                for k in range(1, N_MFCC+1):
                    feat_colNames.append("MFCC_" + str(k) + "_mc")
                for k in range(1, N_MFCC+1):
                    feat_colNames.append("MFCC_D_" + str(k) + "_mc")
                for k in range(1, N_MFCC+1):
                    feat_colNames.append("MFCC_2D_" + str(k) + "_mc")
                
                for k in range(1, N_MFCC+1):
                    feat_colNames.append("MFCC_" + str(k) + "_st1")
                for k in range(1, N_MFCC+1):
                    feat_colNames.append("MFCC_D_" + str(k) + "_st1")
                for k in range(1, N_MFCC+1):
                    feat_colNames.append("MFCC_2D_" + str(k) + "_st1")
                
                for k in range(1, N_MFCC+1):
                    feat_colNames.append("MFCC_" + str(k) + "_st2")
                for k in range(1, N_MFCC+1):
                    feat_colNames.append("MFCC_D_" + str(k) + "_st2")
                for k in range(1, N_MFCC+1):
                    feat_colNames.append("MFCC_2D_" + str(k) + "_st2")
                
                feat_colNames.append('TB_status')
                
                df = pd.DataFrame(columns = feat_colNames)
                
                for pat in patients_info:
                    
                    print ("Getting feature vectors for: ", pat)
                    
                    print('N_MFCC ', N_MFCC)
                    print('N_frame ', N_frame)
                    print('No. of Bins ', B)
                    print('Avg ', Avg)
                    
                    
                    ## For main mic: 
                    # Calculate CMS mean
                    all_cough_audio_mc = []
                    all_cough_audio_st_1 = []
                    all_cough_audio_st_2 = []
                    # for c in patients_info[pat]:
                    #     print(c)
                    for i in range(len(patients_info[pat])-2):
                        # print(len(patients_info[pat]['cough_'+str(i)]['audio']))
                        all_cough_audio_mc.extend(patients_info[pat]['cough_'+str(i)]['audio_mc'])
                        all_cough_audio_st_1.extend(patients_info[pat]['cough_'+str(i)]['audio_st1'])
                        all_cough_audio_st_2.extend(patients_info[pat]['cough_'+str(i)]['audio_st2'])
                    
                    mfcc_means_mc = get_mfcc_means(all_cough_audio_mc, N_frame, N_frame, N_MFCC)
                    mfcc_means_st_1 = get_mfcc_means(all_cough_audio_st_1, N_frame, N_frame, N_MFCC)
                    mfcc_means_st_2 = get_mfcc_means(all_cough_audio_st_2, N_frame, N_frame, N_MFCC)
                    
                    
                    
                    for i in range(len(patients_info[pat])-2):
                        # print(len(patients_info[pat]['cough_'+str(i)]['audio']))
                        # print( patients_info[pat]['cough_'+str(i)] )
                        # print(i)
                        # features_vectors[pat]['cough_'+str(i)] = {}
                        
                        # features_vectors_MFCC[pat][str(N_MFCC)][str(N_frame)][str(N_seg)]['cough_'+str(i)] = {}
                        
                        # This is a cough 
                        patients_info[pat]['cough_'+str(i)]['audio_mc']
                        patients_info[pat]['cough_'+str(i)]['audio_st1']
                        patients_info[pat]['cough_'+str(i)]['audio_st2']
                        
                        ## For main mic:
                        wav_file = patients_info[pat]['cough_'+str(i)]['audio_mc']
                        fs = patients_info[pat]['cough_'+str(i)]['sr_rate']
                        if len(wav_file) < N_frame :
                            continue
                        # frame audio
                        frames = librosa.util.frame(wav_file, frame_length = N_frame, hop_length = N_frame)
                        """
                        Get which frames go into which bins
                        """
                        bins, empty_bin_flag = get_binned_framenums(wav_file, frames.shape[1], B)
                        """ If one of the bins are empty, just leave out this frame """
                        if empty_bin_flag:
                            print('\n\nSomething is very wrong with bin number calculations \n\n')
                            continue
                        
                        MFCC_mc, MFCC_D_mc, MFCC_A_mc = calc_MFCC_D_A(wav_file.astype(np.float32), mfcc_means_mc, N_frame, N_frame, N_MFCC)
                        if len(MFCC_D_mc) == 0 or len(MFCC_A_mc) == 0:
                            print('The MFCC shape is not big enough. Skipping ... ')
                            continue
                        
                        ## For stethoscope 1:
                        wav_file = patients_info[pat]['cough_'+str(i)]['audio_st1']
                        if len(wav_file) < N_frame :
                            continue
                        # frame audio
                        frames = librosa.util.frame(wav_file, frame_length = N_frame, hop_length = N_frame)
                        """
                        Get which frames go into which bins
                        """
                        bins, empty_bin_flag = get_binned_framenums(wav_file, frames.shape[1], B)
                        """ If one of the bins are empty, just leave out this frame """
                        if empty_bin_flag:
                            print('\n\nSomething is very wrong with bin number calculations \n\n')
                            continue
                        MFCC_st_1, MFCC_D_st_1, MFCC_A_st_1 = calc_MFCC_D_A(wav_file.astype(np.float32), mfcc_means_st_1, N_frame, N_frame, N_MFCC)
                        if len(MFCC_D_st_1) == 0 or len(MFCC_A_st_1) == 0:
                            print('The MFCC shape is not big enough. Skipping ... ')
                            continue
                            
                        ## For stethoscope 2: 
                        wav_file = patients_info[pat]['cough_'+str(i)]['audio_st2']
                        if len(wav_file) < N_frame :
                            continue
                        # frame audio
                        frames = librosa.util.frame(wav_file, frame_length = N_frame, hop_length = N_frame)
                        """
                        Get which frames go into which bins
                        """
                        bins, empty_bin_flag = get_binned_framenums(wav_file, frames.shape[1], B)
                        """ If one of the bins are empty, just leave out this frame """
                        if empty_bin_flag:
                            print('\n\nSomething is very wrong with bin number calculations \n\n')
                            continue
                        MFCC_st_2, MFCC_D_st_2, MFCC_A_st_2 = calc_MFCC_D_A(wav_file.astype(np.float32), mfcc_means_st_2, N_frame, N_frame, N_MFCC)
                        if len(MFCC_D_st_2) == 0 or len(MFCC_A_st_2) == 0:
                            print('The MFCC shape is not big enough. Skipping ... ')
                            continue
                        
                        MFCC_vec = np.vstack((MFCC_mc, MFCC_D_mc, MFCC_A_mc, MFCC_st_1, MFCC_D_st_1, MFCC_A_st_1, MFCC_st_2, MFCC_D_st_2, MFCC_A_st_2)).transpose()
                        
                        """
                        Extract features from each frame
                        Save all features for complete dataset
                        And put feature vecs into correct bins
                        """
                        
                        feature_vecs = []
                        
                        # for each frame
                        for k in range(frames.shape[1]):
                
                            frame = frames[:,k]
                
                            # Apply a hamming window
                            frame = frame * hamming(len(frame))
                
                            zcr = zero_crossing_rate(frame)
                            kurt = kurtosis(frame)
                            logE = LogEnergy(frame)
                
                            vec = [pat,i,k,kurt,zcr,logE]
                            vec.extend(MFCC_vec[k,:])
                            # Add meta_data
                            vec.append(patients_info[pat]['TB_status'])
                
                            """
                            Check which bin this frame is in
                            """
                            bin_no = 1
                            if B > 1:
                                for b in bins:
                                    if k in b:
                                        break
                                    else:
                                        bin_no +=1
                
                            vec.insert(1,bin_no)
                            feature_vecs.append(vec)
                        
                        feature_matrix = np.vstack(feature_vecs)
                        feature_vecs = []
                        MFCC_vec = []
                        temp_df = pd.DataFrame(feature_matrix, columns = feat_colNames)
                        feature_matrix = []
                        
                        """
                        We need to duplicate the temp_df for each bin
                        """
                        if B > frames.shape[1]:
                            for b in range(1,B+1):
                                temp_df['Bin_No'] = b
                                df = df.append(temp_df)
                                
                        else:
                            df = df.append(temp_df)
                        temp_df = []
                    
                    
                    # if (len(df.groupby('Study_Num')['Cough_No'].unique()[0]) != len(patients_info[pat])-1):
                    #     print('pat', pat, ' has coughs: ', (len(patients_info[pat])-1))
                    #     print('df shape ', df.shape)
                    #     print('Feature has been extracted for coughs: ', len(df.groupby('Study_Num')['Cough_No'].unique()[0]))
                    #     print('Number of coughs missing in the feature extraction: ', (len(patients_info[pat])-1)- len(df.groupby('Study_Num')['Cough_No'].unique()[0]))
                        
                # Convert numeric columns to the appropriate dtypes
                df = df.apply(lambda x: pd.to_numeric(x, errors = 'ignore'))
                
                if Avg == 1:    
                    df = average_over_bins(df, True)
                    fname = 'features_MFCC='+ str(N_MFCC) + '_Frame=' +str(N_frame)+ '_B=' +str(B)+'_Avg=True.csv'
                elif Avg == 0:
                    fname = 'features_MFCC='+ str(N_MFCC) + '_Frame=' +str(N_frame)+ '_B=' +str(B)+'_Avg=False.csv'
                
                if len(df.Study_Num.unique()) == len(patients_info):
                    ## Save the feature as csv file
                    df.to_csv('../data/feature_dataset/MFCC_features/'+fname,sep=';',index=False,index_label=False)
                else:
                    print('The dataset does not contain all patients.')
                df = []



'''
Now, extract features for filterbanks 
'''
for N_FBANK in tqdm(N_FBANKS):
    for N_frame in N_frames:
        for B in Bs:
            for Avg in Avgs:
                
                feat_colNames = ["Study_Num",'Bin_No','Cough_No','Win_No',"ZCR","Kurtosis","LogE"]
                for k in range(1,N_FBANK+1):
                    feat_colNames.append('FBANK_'+ str(k)+ '_mc')
                for k in range(1,N_FBANK+1):
                    feat_colNames.append('FBANK_'+ str(k)+ '_st1')
                for k in range(1,N_FBANK+1):
                    feat_colNames.append('FBANK_'+ str(k)+ '_st2')
                feat_colNames.append('TB_status')
                
                df = pd.DataFrame(columns = feat_colNames)
                
                for pat in patients_info:
                    
                    print ("Getting feature vectors for: ", pat)
                    
                    print('N_FBANK ', N_FBANK)
                    print('N_frame ', N_frame)
                    print('No. of Bins ', B)
                    print('Avg ', Avg)
                    
                    for i in range(len(patients_info[pat])-1):
                        
                        # This is a cough 
                        patients_info[pat]['cough_'+str(i)]['audio_mc']
                        patients_info[pat]['cough_'+str(i)]['audio_st1']
                        patients_info[pat]['cough_'+str(i)]['audio_st2']
                        
                        ## For the main mic 
                        wav_file = patients_info[pat]['cough_'+str(i)]['audio_mc']
                        fs = patients_info[pat]['cough_'+str(i)]['sr_rate']
                        if len(wav_file) < N_frame :
                            continue
                        # frame audio
                        frames = librosa.util.frame(wav_file, frame_length = N_frame, hop_length = N_frame)
                        """
                        Get which frames go into which bins
                        """
                        bins, empty_bin_flag = get_binned_framenums(wav_file, frames.shape[1], B)
                        """ If one of the bins are empty, just leave out this frame """
                        if empty_bin_flag:
                            print('\n\nSomething is very wrong with bin number calculations \n\n')
                            continue
                        """
                        Calculate the Filterbanks
                        This is the frequency spectrum binned into
                        N_FBANKS number of 'filterbanks', which are
                        triangular filters linearly spaced on the 
                        frequency axis
                        """
                        FBANKS_mc = log_filterbanks(wav_file,
                                                nfilters 	= N_FBANK,
                                                fs 		= fs,
                                                winlen 	= N_frame,
                                                winstep 	= N_frame,
                                                nfft 		= N_frame,
                                                spec 		= 'pow')
                        
                        
                        ## For the stethoscope 1
                        wav_file = patients_info[pat]['cough_'+str(i)]['audio_st1']
                        if len(wav_file) < N_frame :
                            continue
                        # frame audio
                        frames = librosa.util.frame(wav_file, frame_length = N_frame, hop_length = N_frame)
                        """
                        Get which frames go into which bins
                        """
                        bins, empty_bin_flag = get_binned_framenums(wav_file, frames.shape[1], B)
                        """ If one of the bins are empty, just leave out this frame """
                        if empty_bin_flag:
                            print('\n\nSomething is very wrong with bin number calculations \n\n')
                            continue
                        """
                        Calculate the Filterbanks
                        This is the frequency spectrum binned into
                        N_FBANKS number of 'filterbanks', which are
                        triangular filters linearly spaced on the 
                        frequency axis
                        """
                        FBANKS_st_1 = log_filterbanks(wav_file,
                                                nfilters 	= N_FBANK,
                                                fs 		= fs,
                                                winlen 	= N_frame,
                                                winstep 	= N_frame,
                                                nfft 		= N_frame,
                                                spec 		= 'pow')
                        
                        
                        ## For the stethoscope 2
                        wav_file = patients_info[pat]['cough_'+str(i)]['audio_st2']
                        if len(wav_file) < N_frame :
                            continue
                        # frame audio
                        frames = librosa.util.frame(wav_file, frame_length = N_frame, hop_length = N_frame)
                        """
                        Get which frames go into which bins
                        """
                        bins, empty_bin_flag = get_binned_framenums(wav_file, frames.shape[1], B)
                        """ If one of the bins are empty, just leave out this frame """
                        if empty_bin_flag:
                            print('\n\nSomething is very wrong with bin number calculations \n\n')
                            continue
                        """
                        Calculate the Filterbanks
                        This is the frequency spectrum binned into
                        N_FBANKS number of 'filterbanks', which are
                        triangular filters linearly spaced on the 
                        frequency axis
                        """
                        FBANKS_st_2 = log_filterbanks(wav_file,
                                                nfilters 	= N_FBANK,
                                                fs 		= fs,
                                                winlen 	= N_frame,
                                                winstep 	= N_frame,
                                                nfft 		= N_frame,
                                                spec 		= 'pow')
                        
                        
                        FBANKS = np.concatenate((FBANKS_mc, FBANKS_st_1, FBANKS_st_2), axis=1)
                        
                        """
                        Extract features from each frame
                        Save all features for complete dataset
                        And put feature vecs into correct bins
                        """
                        feature_vecs = []
                        
                        # for each frame
                        for k in range(frames.shape[1]):
                
                            frame = frames[:,k]
                
                            # Apply a hamming window
                            frame = frame * hamming(len(frame))
                
                            zcr = zero_crossing_rate(frame)
                            kurt = kurtosis(frame)
                            logE = LogEnergy(frame)
                
                            vec = [pat,i,k,kurt,zcr,logE]
                            vec.extend(FBANKS[k,:])
                            # Add meta_data
                            vec.append(patients_info[pat]['TB_status'])
                
                            """
                            Check which bin this frame is in
                            """
                            bin_no = 1
                            if B > 1:
                                for b in bins:
                                    if k in b:
                                        break
                                    else:
                                        bin_no +=1
                
                            vec.insert(1,bin_no)
                            feature_vecs.append(vec)
                        
                        feature_matrix = np.vstack(feature_vecs)
                        feature_vecs = []
                        FBANKS = []
                        temp_df = pd.DataFrame(feature_matrix, columns = feat_colNames)
                        feature_matrix = []
                        
                        """
                        We need to duplicate the temp_df for each bin
                        """
                        if B > frames.shape[1]:
                            for b in range(1,B+1):
                                temp_df['Bin_No'] = b
                                df = df.append(temp_df)
                                
                        else:
                            df = df.append(temp_df)
                        temp_df = []
                # Convert numeric columns to the appropriate dtypes
                df = df.apply(lambda x: pd.to_numeric(x, errors = 'ignore'))
                
                if Avg == 1:
                    df = average_over_bins(df, True)
                    fname = 'features_FBANK='+ str(N_FBANK) + '_Frame=' +str(N_frame)+ '_B=' +str(B)+'_Avg=True.csv'
                elif Avg == 0:
                    fname = 'features_FBANK='+ str(N_FBANK) + '_Frame=' +str(N_frame)+ '_B=' +str(B)+'_Avg=False.csv'
                
                if len(df.Study_Num.unique()) == len(patients_info):
                    ## Save the feature as csv file
                    df.to_csv('../data/feature_dataset/FBANK_features/'+fname,sep=';',index=False,index_label=False)
                else:
                    print('The dataset does not contain all patients.')
                
                df = []








