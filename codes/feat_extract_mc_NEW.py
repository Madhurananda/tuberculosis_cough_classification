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

sys.path.insert(0, '/home/madhu/work/delete_later/cough_classification/codes//Renier_scripts')
# from log_filterbanks import log_filterbanks as log_fbanks
from log_filterbanks import log_filterbanks
from helper import *
from feature_extraction_MFCC import calc_mfcc, get_mfcc_means, calc_MFCC_D_A

import warnings
import librosa.display


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


def signaltonoise_dB(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))


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



## Load the dataset ... 
with open ('../data/raw_data/patients_information', 'rb') as fp:
    patients_info = pickle.load(fp)


'''
This is where the recording information is extracted ... 
'''
N_TB = 0
N_NTB = 0

N_TB_cough = []
N_NTB_cough = []

audio_TB = []
audio_NTB = []

list_SNR = []
list_TB_SNR = []
list_NTB_SNR = []



index_count = 0

for key in patients_info:
    
    if key.startswith('TB'):
        N_TB += 1
        a_cough = 0
        for aCough in patients_info[key]:
            if aCough.startswith('cough'):
                cough_audio = patients_info[key][aCough]['audio_mc']
                fs = patients_info[key][aCough]['sr_rate']
                
                audio_TB.append( len(cough_audio)/fs )
                
                a_cough += 1
        
        N_TB_cough.append(a_cough)
        list_TB_SNR.append(signaltonoise_dB(cough_audio, axis = 0, ddof = 0))
        
    elif key.startswith('NTB'):
        N_NTB += 1
        a_cough = 0
        for aCough in patients_info[key]:
            if aCough.startswith('cough'):
                cough_audio = patients_info[key][aCough]['audio_mc']
                fs = patients_info[key][aCough]['sr_rate']
                
                audio_NTB.append( len(cough_audio)/fs )
                a_cough += 1
        
        N_NTB_cough.append(a_cough)
        list_NTB_SNR.append(signaltonoise_dB(cough_audio, axis = 0, ddof = 0))
    list_SNR.append(signaltonoise_dB(cough_audio, axis = 0, ddof = 0))
    
    
    # print('The SNR in the audio is: ', signaltonoise_dB(cough_audio, axis = 0, ddof = 0), ' for index: ', index_count)
    
    if index_count == 47: 
        print('The SNR in the audio is: ', signaltonoise_dB(cough_audio, axis = 0, ddof = 0), ' for index: ', index_count)
        noisy_cough = cough_audio
    elif index_count == 27:
        print('The SNR in the audio is: ', signaltonoise_dB(cough_audio, axis = 0, ddof = 0), ' for index: ', index_count)
        clean_cough = cough_audio
    index_count += 1


print('Number of TB patient: ', N_TB)
print('Number of non-TB patient: ', N_NTB)

print('Average TB cough per patient: ', np.mean(np.array(N_TB_cough)))
print('Average non-TB cough per patient: ', np.mean(np.array(N_NTB_cough)))

print('Average TB cough length: ', np.mean(np.array(audio_TB)), ' sec')
print('Average non-TB cough length: ', np.mean(np.array(audio_NTB)), ' sec')

print('Total TB cough length: ', np.sum(np.array(audio_TB)), ' sec')
print('Average non-TB cough length: ', np.sum(np.array(audio_NTB)), ' sec')


print('TB patients:: The mean SNR is ', np.mean(list_TB_SNR), ' dB')
print('TB patients:: The SD SNR is ', np.std(list_TB_SNR), ' dB')

print('Non-TB patients:: The mean SNR is ', np.mean(list_NTB_SNR), ' dB')
print('Non-TB patients:: The SD SNR is ', np.std(list_NTB_SNR), ' dB')

print('All patients:: The mean SNR is ', np.mean(list_SNR), ' dB')
print('All patients:: The SD SNR is ', np.std(list_SNR), ' dB')



'''
The SNR in the audio is:  -87.36533787438574  for index:  0
The SNR in the audio is:  -66.50286977736273  for index:  1
The SNR in the audio is:  -75.97700571025653  for index:  2
The SNR in the audio is:  -71.53564504370715  for index:  3
The SNR in the audio is:  -67.49264498822964  for index:  4
The SNR in the audio is:  -59.44082759586962  for index:  5
The SNR in the audio is:  -44.83039072808783  for index:  6
The SNR in the audio is:  -65.34376079502404  for index:  7
The SNR in the audio is:  -50.512766930608954  for index:  8
The SNR in the audio is:  -68.29375591739836  for index:  9
The SNR in the audio is:  -75.09384536924607  for index:  10
The SNR in the audio is:  -64.15447989234825  for index:  11
The SNR in the audio is:  -46.21767814690091  for index:  12
The SNR in the audio is:  -58.49781917485097  for index:  13
The SNR in the audio is:  -62.684321127608975  for index:  14
The SNR in the audio is:  -63.20348334990358  for index:  15
The SNR in the audio is:  -65.36041132803936  for index:  16
The SNR in the audio is:  -48.82174498514246  for index:  17
The SNR in the audio is:  -57.85657108827248  for index:  18
The SNR in the audio is:  -56.304281378119015  for index:  19
The SNR in the audio is:  -50.3325059471472  for index:  20
The SNR in the audio is:  -56.88211862170772  for index:  21
The SNR in the audio is:  -59.11754870029676  for index:  22
The SNR in the audio is:  -43.22543312091753  for index:  23
The SNR in the audio is:  -56.36653534344569  for index:  24
The SNR in the audio is:  -44.4166586609217  for index:  25
The SNR in the audio is:  -90.09527017210976  for index:  26
The SNR in the audio is:  -96.0847730949137  for index:  27
The SNR in the audio is:  -46.67828507853805  for index:  28
The SNR in the audio is:  -64.50763362212763  for index:  29
The SNR in the audio is:  -59.66159597939158  for index:  30
The SNR in the audio is:  -51.81766297014473  for index:  31
The SNR in the audio is:  -58.02651953799717  for index:  32
The SNR in the audio is:  -71.0912454665441  for index:  33
The SNR in the audio is:  -75.68055225224781  for index:  34
The SNR in the audio is:  -51.48041917121919  for index:  35
The SNR in the audio is:  -65.50291781299666  for index:  36
The SNR in the audio is:  -45.98559444497292  for index:  37
The SNR in the audio is:  -60.989827623746955  for index:  38
The SNR in the audio is:  -41.14170992221551  for index:  39
The SNR in the audio is:  -68.65341082268137  for index:  40
The SNR in the audio is:  -66.65958658884541  for index:  41
The SNR in the audio is:  -58.38571117812535  for index:  42
The SNR in the audio is:  -55.37335346969371  for index:  43
The SNR in the audio is:  -35.835671180122944  for index:  44
The SNR in the audio is:  -63.7972238677385  for index:  45
The SNR in the audio is:  -74.92002575606926  for index:  46
The SNR in the audio is:  -41.50896175689877  for index:  47
The SNR in the audio is:  -59.091296459291264  for index:  48
The SNR in the audio is:  -63.27549401244438  for index:  49
The SNR in the audio is:  -76.9598735517078  for index:  50
'''

max_index = list_SNR.index(max(list_SNR))

min_index = list_SNR.index(min(list_SNR))

# arr1 = [[20, 2, 7, 1, 34],
#         [50, 12, 12, 34, 4]]
  
# arr2 = [50, 12, 12, 34, 4]

# print ("\nsignaltonoise ratio for arr2 : ", 
#        signaltonoise_dB(arr2, axis = 0, ddof = 0))

from scipy import fftpack

# def draw_audio_spec(audio_data, sr, title):
#     t = np.linspace(0, len(audio_data)/sr, len(audio_data), endpoint=False)
    
#     fig, axs = plt.subplots(3, 1, constrained_layout=True, figsize=(20, 10))
    
#     # axs[0].tick_params(axis='both', which='major', labelsize=15)
#     # axs[0].tick_params(axis='both', which='minor', labelsize=15)
    
#     for ax in axs:
#         ax.tick_params(axis='both', which='major', labelsize=22)
#         ax.tick_params(axis='both', which='minor', labelsize=22)
    
#     # fig, axs = plt.subplots(3, 1, figsize=(20, 10))
#     # axs[0] = plt.subplot(3,1,1)
    
#     # axs[0].plot(t, audio_data, '-')
    
#     y = np.abs(audio_data)
#     norm_y = np.true_divide(audio_data, max(y))
#     morm_y_norm = np.multiply( norm_y, [0.9] )
#     axs[0].plot(t, morm_y_norm, '-')
    
#     # librosa.display.waveshow(audio_data, sr=sr, ax=axs[0])
#     axs[0].set_title('The '+title, fontsize=35)
#     # axs[0].set_title('Acoustic signal showing two successive coughs', fontsize=20)
#     axs[0].set_xlabel('Time (s)', fontsize=26)
#     axs[0].set_ylabel('Amplitude', fontsize=26)
#     axs[0].set_xlim(-0.01, (len(audio_data)/sr)+0.01)
#     # fig.suptitle(title, fontsize=16)
    
#     X = fftpack.fft(audio_data)
#     freqs = fftpack.fftfreq(len(audio_data)) * sr
    
#     y = np.abs(X)
#     norm_y = np.true_divide(y, max(y))
    
#     # axs[1] = plt.subplot(3,1,2)
#     axs[1].plot(freqs, norm_y)
#     axs[1].set_xlabel('Frequency (Hz)', fontsize=26)
#     axs[1].set_ylabel('Spectral Magnitude', fontsize=26)
#     axs[1].set_xlim(-0.01, sr / 2)
#     # axs[1].set_title('Average spectrum of the two coughs', fontsize=20)
#     axs[1].set_title('Average spectrum of the '+title, fontsize=32)
    
#     M = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128, hop_length=128)
#     # M = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
#     S_dB = librosa.power_to_db(M, ref=np.max)
    
#     # axs[2] = plt.subplot(3,1,3)
#     axs[2].set_xlabel('Time (sec)', fontsize=26)
#     axs[2].set_ylabel('Frequency (Hz)', fontsize=26)
#     # axs[2].set_xticklabels([0, 1], Fontsize=10 )
#     # axs[2].set_xticklabels(np.linspace(0, len(audio_data)/sr, 5))
#     # axs[2].set_title('Spectrogram representation for the two coughs', fontsize=20)
#     axs[2].set_title('Spectrogram representation of the '+title, fontsize=32)
#     librosa.display.specshow(S_dB, x_axis='s', y_axis='mel', sr=sr, ax=axs[2], x_coords=np.linspace(-0.1, (len(audio_data)/sr)+0.01, S_dB.shape[1]))
#     plt.show()



# draw_audio_spec(noisy_cough, fs, "noisy cough")
# # draw_audio_spec(noisy_cough[int(0.02*fs):int(0.5*fs)], fs, "Noisy Cough")

# draw_audio_spec(clean_cough, fs, "clean cough")





'''
This is to visualise the TB Wallacedene dataset which is used in the publication 
'''

# df = pd.read_csv("/home/madhu/work/Publications/TB Cough Classification/excel_files/WallacedeneUltra-MarisaTBCasesVSSuspe_DATA_LABELS_2021-07-14_1653.csv")

# ## Must drop the subjects whose cough lengths are zero
# # 'pat_list_zero' is generated later. 
# for i in to_drop_pats:
#     new_df = new_df[new_df.id != i]



# # df = df[df.uuid != '50e0d0d3-a060-4ee2-b85c-a659e3bb394f']
# # df = df[df.uuid != 'ee4a0435-204c-44cd-8d2d-4dc83faf18b3']
# # df = df[df.uuid != '848cffae-f523-44b8-9577-53723e7e0d1c']

# df[(df['covid_status'] == 'healthy')]

# df[(df['covid_status'] == 'healthy') | (df['covid_status'] == 'positive_mild')]

# df[(df['covid_status'] == 'healthy') | (df['covid_status'] == 'positive_mild') | (df['covid_status'] == 'positive_asymp') | (df['covid_status'] == 'positive_moderate')]


# plt.style.use('ggplot')

# new_df = new_df.replace(np.nan,0)

# df_gender = new_df.groupby(['g'])['id'].count()

# df_gender.to_csv("../data/feature_data/gender.csv")

# # # df_gender.plot.bar()
# # ax = df_gender.plot(kind='bar', figsize=(15, 8), color="red", fontsize=12, rot=90);
# # ax.set_alpha(0.8)
# # ax.set_title("Age Distribution", fontsize=22)
# # # ax.set_xticks([5, 18, 20, 25, 60, 81])
# # ax.set_ylabel("Some Heading on Y-Axis", fontsize=15);

# ax = df_gender.plot(kind='bar', figsize=(20,10), color="indigo", fontsize=20, rot=0);
# ax.set_alpha(0.8)
# ax.set_title("Gender Distribution", fontsize=25)
# ax.legend(['Number of Subjects'], fontsize=20)
# # create a list to collect the plt.patches data
# totals = []
# # find the values and append to list
# for i in ax.patches:
#     totals.append(i.get_height())
# # set individual bar lables using above list
# total = sum(totals)
# # set individual bar lables using above list
# for i in ax.patches:
#     # get_x pulls left or right; get_height pushes up or down
#     ax.text(i.get_x()+.20, i.get_height()-60, \
#             str(round((i.get_height()/total)*100, 2))+'%', fontsize=22,
#                 color='white')
#     ax.text(i.get_x()+.20, i.get_height()+10, \
#             str(i.get_height()), fontsize=22,
#                 color='black')






# df_country = new_df.groupby(['l_c'])['id'].count()

# df_country.to_csv("../data/feature_data/country.csv")

# print('Total number of participants: ', df_country.sum())

# df_country = round((df_country/df_country.sum())*100, 2)



# index = ['Asia', 'Australia', 'Europe', 'North America', 'South America']
# new_df_country = pd.DataFrame({'number of subjects': [1090, 2, 28, 50, 1]}, index=index)

# new_df_country.to_csv("../data/feature_data/continents.csv")

# import pylab as plot
# params = {'legend.fontsize': 20,
#           'legend.handlelength': 2}
# plot.rcParams.update(params)

# # ax = new_df_country.plot(kind='pie', figsize=(20,10), fontsize=20, y='number of subjects', autopct='%1.1f%%');
# # # ax = new_df_country.plot(kind='pie', figsize=(20,10), fontsize=20, y='number of subjects', autopct='%.2f');
# # ax.set_alpha(0.8)
# # ax.set_title("Origin distribution of the Participants", fontsize=25)
# # ax.set_xticks([0, 2])

# # # create a list to collect the plt.patches data
# # totals = []

# # # find the values and append to list
# # for i in ax.patches:
# #     totals.append(i.get_width())

# # # set individual bar lables using above list
# # total = sum(totals)

# # # set individual bar lables using above list
# # for i in ax.patches:
# #     # get_width pulls left or right; get_y pushes up or down
# #     ax.text(i.get_width()+.3, i.get_y()+.38, \
# #             str(round((i.get_width()/total)*100, 2))+'%', fontsize=15,
# # color='dimgrey')

# # # invert for largest on top 
# # ax.invert_yaxis()





# ## Do nothing as it will be shown in a table

# # # df_country.plot.bar()
# # ax = df_country.plot(kind='bar', figsize=(15, 8), color="indigo", fontsize=12, rot=90);
# # ax.set_alpha(0.8)
# # ax.set_title("Age Distribution", fontsize=22)
# # # ax.set_xticks([5, 18, 20, 25, 60, 81])
# # ax.set_ylabel("Some Heading on Y-Axis", fontsize=15);


# df_cough = new_df.groupby(['covid_status'])['id'].count()

# df_cough = df_cough.drop('resp_illness_not_identified')
# df_cough = df_cough.drop('recovered_full')
# df_cough = df_cough.drop('no_resp_illness_exposed')

# index = ['Healthy', 'COVID Positive']
# new_df_cough = pd.DataFrame({'number of subjects': [1079, 92]}, index=index)

# new_df_cough.to_csv("../data/feature_data/covid_stat.csv")



# ax = new_df_cough.plot(kind='bar', figsize=(20,10), color="indigo", fontsize=20, rot=0);
# ax.set_alpha(0.8)
# ax.set_title("COVID and Healthy Subjects Distribution", fontsize=25)
# ax.legend(['Number of Subjects'], fontsize=20)
# # create a list to collect the plt.patches data
# totals = []
# # find the values and append to list
# for i in ax.patches:
#     totals.append(i.get_height())
# # set individual bar lables using above list
# total = sum(totals)
# # set individual bar lables using above list
# for i in ax.patches:
#     # get_x pulls left or right; get_height pushes up or down
#     ax.text(i.get_x()+.20, i.get_height()-60, \
#             str(round((i.get_height()/total)*100, 2))+'%', fontsize=22,
#                 color='white')
#     ax.text(i.get_x()+.20, i.get_height()+10, \
#             str(i.get_height()), fontsize=22,
#                 color='black')






# # # df_cough.plot.bar(y='Health Status of participants', rot = 0)
# # ax = new_df_cough.plot(kind='bar', figsize=(20, 10), color="red", fontsize=20, rot=0);
# # ax.set_alpha(0.8)
# # ax.set_title("COVID status", fontsize=22)
# # ax.set_ylabel("Some Heading on Y-Axis", fontsize=15);
# # # ax.set_xlabel("COVID status", fontsize=15);




# df_gender = df.groupby(['Sex'])['Subject_ID'].count()
# df_gender.to_csv("/home/madhu/work/Publications/TB Cough Classification/excel_files/gender.csv")



# df_age = df.groupby(['Age'])['Subject_ID'].count()
# df_age.to_csv("/home/madhu/work/Publications/TB Cough Classification/excel_files/age.csv")



# index = ['TB', 'non-TB']
# df_TB = pd.DataFrame({'Healthy Status': [16, 35]}, index=index)

# df_TB.to_csv("/home/madhu/work/Publications/TB Cough Classification/excel_files/health.csv")










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


N_MFCCs = [26]
N_frames = [2**11]
Bs = [1]
Avgs = [1]


# N_MFCCs = list(range(13, (13*5)+1, 13))
# # N_MFCCs = list(range(13*6, (13*9)+1, 13))
# # N_MFCCs = [26]

# N_FBANKS = list(range(40, 210, 20))
# # N_FBANKS = list(range(40, 210, 20)) + list(range(300, 1001, 100))
# N_frames = [2**9, 2**10, 2**11, 2**12] # This should be equal to M and N
# # N_frames = [2**11] # This should be equal to M and N
# # Bs = range(1, 5, 1)
# Bs = [1]
# Avgs = range(2)




for N_MFCC in tqdm(N_MFCCs):
    for N_frame in N_frames:
        for B in Bs:
            for Avg in Avgs:
                
                feat_colNames = ["Study_Num",'Bin_No','Cough_No','Win_No',"ZCR","Kurtosis","LogE"]
                for k in range(1, N_MFCC+1):
                    feat_colNames.append("MFCC_" + str(k) + "_mc")
                for k in range(1, N_MFCC+1):
                    feat_colNames.append("MFCC_D_" + str(k) + "_mc")
                for k in range(1, N_MFCC+1):
                    feat_colNames.append("MFCC_2D_" + str(k) + "_mc")
                
                
                feat_colNames.append('TB_status')
                
                df = pd.DataFrame(columns = feat_colNames)
                
                for pat in patients_info:
                    
                    # if pat != 'NTB_Wu0376':
                    #     continue

                    print ("Getting feature vectors for: ", pat)
                    
                    print('N_MFCC ', N_MFCC)
                    print('N_frame ', N_frame)
                    print('No. of Bins ', B)
                    print('Avg ', Avg)
                    
                    
                    ## For main mic: 
                    # Calculate CMS mean
                    all_cough_audio_mc = []
                    for i in range(0, len(patients_info[pat])-2):
                        all_cough_audio_mc.extend(patients_info[pat]['cough_'+str(i)]['audio_mc'])
                    
                    mfcc_means_mc = get_mfcc_means(all_cough_audio_mc, N_frame, N_frame, N_MFCC, patients_info['TB_Wu0378']['cough_0']['sr_rate'])
                    
                    for i in range(len(patients_info[pat])-2):
                        # print(len(patients_info[pat]['cough_'+str(i)]['audio']))
                        # print( patients_info[pat]['cough_'+str(i)] )
                        # print(i)
                        # features_vectors[pat]['cough_'+str(i)] = {}
                        
                        # features_vectors_MFCC[pat][str(N_MFCC)][str(N_frame)][str(N_seg)]['cough_'+str(i)] = {}
                        
                        # This is a cough 
                        patients_info[pat]['cough_'+str(i)]['audio_mc']
                        
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
                        
                        MFCC_mc, MFCC_D_mc, MFCC_A_mc = calc_MFCC_D_A(wav_file.astype(np.float32), mfcc_means_mc, N_frame, N_frame, N_MFCC, fs)
                        if len(MFCC_D_mc) == 0 or len(MFCC_A_mc) == 0:
                            print('The MFCC shape is not big enough. Skipping ... ')
                            continue
                        
                        
                        MFCC_vec = np.vstack((MFCC_mc, MFCC_D_mc, MFCC_A_mc)).transpose()
                        
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

                            # print('cough no:', i)
                            # print('zcr: ', zcr)
                            # print('kurt: ', kurt)
                            # print('logE: ', logE)
                
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
                        
                # Convert numeric columns to the appropriate dtypes
                df = df.apply(lambda x: pd.to_numeric(x, errors = 'ignore'))
                
                if Avg == 1:    
                    df = average_over_bins(df, True)
                    fname = 'mc_features_MFCC='+ str(N_MFCC) + '_Frame=' +str(N_frame)+ '_B=' +str(B)+'_Avg=True.csv'
                elif Avg == 0:
                    fname = 'mc_features_MFCC='+ str(N_MFCC) + '_Frame=' +str(N_frame)+ '_B=' +str(B)+'_Avg=False.csv'
                
                # df.to_csv('../data/feature_data/features_dataset/'+fname,sep=';',index=False,index_label=False)
                if len(df.Study_Num.unique()) == len(patients_info):
                    ## Save the feature as csv file
                    df.to_csv('../data/feature_data/features_dataset/'+fname,sep=';',index=False,index_label=False)
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
                    feat_colNames.append('FBANK_D_'+ str(k)+ '_mc')
                for k in range(1,N_FBANK+1):
                    feat_colNames.append('FBANK_2D_'+ str(k)+ '_mc')
                feat_colNames.append('TB_status')
                
                df = pd.DataFrame(columns = feat_colNames)
                
                for pat in patients_info:
                    
                    print ("Getting feature vectors for: ", pat)
                    
                    print('N_FBANK ', N_FBANK)
                    print('N_frame ', N_frame)
                    print('No. of Bins ', B)
                    print('Avg ', Avg)
                    
                    
                    ## For main mic: 
                    # Calculate CMS mean
                    all_cough_audio_mc = []
                    for i in range(0, len(patients_info[pat])-2):
                        all_cough_audio_mc.extend(patients_info[pat]['cough_'+str(i)]['audio_mc'])
                    
                    FBANK_means_mc = np.mean( log_filterbanks(np.array(all_cough_audio_mc), nfilters = N_FBANK, fs = patients_info['TB_Wu0378']['cough_0']['sr_rate'], winlen = N_frame, winstep = N_frame, nfft = N_frame, spec = 'pow'), axis=0)
                    # get_mfcc_means(all_cough_audio_mc, N_frame, N_frame, N_MFCC, patients_info['TB_Wu0378']['cough_0']['sr_rate'])
                    np.mean(FBANK_means_mc,axis=0)
                    
                    
                    for i in range(len(patients_info[pat])-2):
                        
                        # This is a cough 
                        patients_info[pat]['cough_'+str(i)]['audio_mc']
                        
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
                        
                        FBANKS_mc_final = FBANKS_mc - FBANK_means_mc
                        
                        if FBANKS_mc_final.shape[1] % 2 == 0:
                            w = FBANKS_mc_final.shape[1] - 1
                        else:
                            w = FBANKS_mc_final.shape[1]
                        
                        if w<3:
                            print('Filterbank shape is: ', FBANKS_mc_final.shape, ' and thus no further features have been extracted for cough no.', i)
                            MFCC_D = []
                            MFCC_A = []
                        else:
                            FBANKS_mc_D = librosa.feature.delta(FBANKS_mc_final, width=w)
                            FBANKS_mc_A = librosa.feature.delta(FBANKS_mc_final, width=w, order=2)
                        
                        if len(FBANKS_mc_D) == 0 or len(FBANKS_mc_A) == 0:
                            print('The FBANK shape is not big enough. Skipping ... ')
                            continue
                        
                        FBANKS_vec_mc = np.hstack(( FBANKS_mc_final, FBANKS_mc_D, FBANKS_mc_A ))
                        
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
                            vec.extend(FBANKS_vec_mc[k,:])
                            # vec.extend(FBANKS_mc[k,:])
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
                        FBANKS_mc = []
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
                    fname = 'mc_features_FBANK='+ str(N_FBANK) + '_Frame=' +str(N_frame)+ '_B=' +str(B)+'_Avg=True.csv'
                elif Avg == 0:
                    fname = 'mc_features_FBANK='+ str(N_FBANK) + '_Frame=' +str(N_frame)+ '_B=' +str(B)+'_Avg=False.csv'
                
                if len(df.Study_Num.unique()) == len(patients_info):
                    ## Save the feature as csv file
                    df.to_csv('../data/feature_data/features_dataset/'+fname,sep=';',index=False,index_label=False)
                else:
                    print('The dataset does not contain all patients.')
                
                df = []












# '''
# Now, extract features for filterbanks 
# '''
# for N_FBANK in tqdm(N_FBANKS):
#     for N_frame in N_frames:
#         for B in Bs:
#             for Avg in Avgs:
                
#                 feat_colNames = ["Study_Num",'Bin_No','Cough_No','Win_No',"ZCR","Kurtosis","LogE"]
#                 for k in range(1,N_FBANK+1):
#                     feat_colNames.append('FBANK_'+ str(k)+ '_mc')
#                 feat_colNames.append('TB_status')
                
#                 df = pd.DataFrame(columns = feat_colNames)
                
#                 for pat in patients_info:
                    
#                     print ("Getting feature vectors for: ", pat)
                    
#                     print('N_FBANK ', N_FBANK)
#                     print('N_frame ', N_frame)
#                     print('No. of Bins ', B)
#                     print('Avg ', Avg)
                    
#                     for i in range(len(patients_info[pat])-2):
                        
#                         # This is a cough 
#                         patients_info[pat]['cough_'+str(i)]['audio_mc']
                        
#                         ## For the main mic 
#                         wav_file = patients_info[pat]['cough_'+str(i)]['audio_mc']
#                         fs = patients_info[pat]['cough_'+str(i)]['sr_rate']
#                         if len(wav_file) < N_frame :
#                             continue
#                         # frame audio
#                         frames = librosa.util.frame(wav_file, frame_length = N_frame, hop_length = N_frame)
#                         """
#                         Get which frames go into which bins
#                         """
#                         bins, empty_bin_flag = get_binned_framenums(wav_file, frames.shape[1], B)
#                         """ If one of the bins are empty, just leave out this frame """
#                         if empty_bin_flag:
#                             print('\n\nSomething is very wrong with bin number calculations \n\n')
#                             continue
#                         """
#                         Calculate the Filterbanks
#                         This is the frequency spectrum binned into
#                         N_FBANKS number of 'filterbanks', which are
#                         triangular filters linearly spaced on the 
#                         frequency axis
#                         """
#                         FBANKS_mc = log_filterbanks(wav_file,
#                                                 nfilters 	= N_FBANK,
#                                                 fs 		= fs,
#                                                 winlen 	= N_frame,
#                                                 winstep 	= N_frame,
#                                                 nfft 		= N_frame,
#                                                 spec 		= 'pow')
                        
                        
#                         """
#                         Extract features from each frame
#                         Save all features for complete dataset
#                         And put feature vecs into correct bins
#                         """
#                         feature_vecs = []
                        
#                         # for each frame
#                         for k in range(frames.shape[1]):
                
#                             frame = frames[:,k]
                
#                             # Apply a hamming window
#                             frame = frame * hamming(len(frame))
                
#                             zcr = zero_crossing_rate(frame)
#                             kurt = kurtosis(frame)
#                             logE = LogEnergy(frame)
                
#                             vec = [pat,i,k,kurt,zcr,logE]
#                             vec.extend(FBANKS_mc[k,:])
#                             # Add meta_data
#                             vec.append(patients_info[pat]['TB_status'])
                
#                             """
#                             Check which bin this frame is in
#                             """
#                             bin_no = 1
#                             if B > 1:
#                                 for b in bins:
#                                     if k in b:
#                                         break
#                                     else:
#                                         bin_no +=1
                
#                             vec.insert(1,bin_no)
#                             feature_vecs.append(vec)
                        
#                         feature_matrix = np.vstack(feature_vecs)
#                         feature_vecs = []
#                         FBANKS_mc = []
#                         temp_df = pd.DataFrame(feature_matrix, columns = feat_colNames)
#                         feature_matrix = []
                        
#                         """
#                         We need to duplicate the temp_df for each bin
#                         """
#                         if B > frames.shape[1]:
#                             for b in range(1,B+1):
#                                 temp_df['Bin_No'] = b
#                                 df = df.append(temp_df)
                                
#                         else:
#                             df = df.append(temp_df)
#                         temp_df = []
#                 # Convert numeric columns to the appropriate dtypes
#                 df = df.apply(lambda x: pd.to_numeric(x, errors = 'ignore'))
                
#                 if Avg == 1:
#                     df = average_over_bins(df, True)
#                     fname = 'mc_features_FBANK='+ str(N_FBANK) + '_Frame=' +str(N_frame)+ '_B=' +str(B)+'_Avg=True.csv'
#                 elif Avg == 0:
#                     fname = 'mc_features_FBANK='+ str(N_FBANK) + '_Frame=' +str(N_frame)+ '_B=' +str(B)+'_Avg=False.csv'
                
#                 if len(df.Study_Num.unique()) == len(patients_info):
#                     ## Save the feature as csv file
#                     df.to_csv('../data/feature_data/features_dataset/'+fname,sep=';',index=False,index_label=False)
#                 else:
#                     print('The dataset does not contain all patients.')
                
#                 df = []




# '''
# This section calculates the PPV and NPV
# '''

# def calc_PPV_NPV(acc, sen):
    
#     spec = 2*acc - sen
    
#     PPV = sen/(sen + (100-spec))
    
#     NPV = spec/(spec + (100-sen))
    
#     return round(PPV*100, 2), round(NPV*100, 2)
    

# acc = 68.71
# sen = 72.10
# print('PPV, NPV: ', calc_PPV_NPV(acc, sen))










