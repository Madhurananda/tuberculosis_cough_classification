#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 19:26:11 2020

@author: madhu
"""

## This script analyses the frequency and other spectral information 


import librosa
import matplotlib.pyplot as plt
import numpy as np

from librosa import display

from scipy import fftpack
from scipy import signal



def draw_audio_spec(audio_data, sr, title):
    
    t = np.linspace(0, len(audio_data)/sr, len(audio_data), endpoint=False)
    
    fig, axs = plt.subplots(3, 1, constrained_layout=True, figsize=(20, 10))
    # fig, axs = plt.subplots(3, 1, figsize=(20, 10))
    # axs[0] = plt.subplot(3,1,1)
    axs[0].plot(t, audio_data, '-')
    axs[0].set_title('A short voice', fontsize=20)
    # axs[0].set_title('Acoustic signal showing two successive coughs', fontsize=20)
    axs[0].set_xlabel('Time (sec)', fontsize=16)
    axs[0].set_ylabel('Amplitude', fontsize=16)
    axs[0].set_xlim(-0.01, (len(audio_data)/sr)+0.01)
    # fig.suptitle(title, fontsize=16)
    
    X = fftpack.fft(audio_data)
    freqs = fftpack.fftfreq(len(audio_data)) * sr
    
    # axs[1] = plt.subplot(3,1,2)
    axs[1].plot(freqs, np.abs(X))
    axs[1].set_xlabel('Frequency (Hz)', fontsize=16)
    axs[1].set_ylabel('Spectral Magnitude', fontsize=16)
    axs[1].set_xlim(-0.01, sr / 2)
    # axs[1].set_title('Average spectrum of the two coughs', fontsize=20)
    axs[1].set_title('Average spectrum of the voice', fontsize=20)
    
    M = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128, hop_length=128)
    # M = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(M, ref=np.max)
    
    # axs[2] = plt.subplot(3,1,3)
    axs[2].set_xlabel('Time (sec)', fontsize=16)
    axs[2].set_ylabel('Frequency (Hz)', fontsize=16)
    # axs[2].set_xticklabels(np.linspace(0, len(audio_data)/sr, 5))
    # axs[2].set_title('Spectrogram representation for the two coughs', fontsize=20)
    axs[2].set_title('Spectrogram representation of the voice', fontsize=20)
    librosa.display.specshow(S_dB, x_axis='Time (sec)', y_axis='mel', sr=sr, ax=axs[2], x_coords=np.linspace(-0.01, (len(audio_data)/sr)+0.01, S_dB.shape[1]))
    plt.show()





audio_data_cough_mc, sr = librosa.load('/home/madhu/work/cough_classification/data/sorted_recording/Wu0376/coughs_mc/cough_10.wav')

audio_data_cough_st_1, sr = librosa.load('/home/madhu/work/cough_classification/data/sorted_recording/Wu0376/coughs_st-1/cough_10.wav')

audio_data_cough_st_2, sr = librosa.load('/home/madhu/work/cough_classification/data/sorted_recording/Wu0376/coughs_st-2/cough_10.wav')

audio_data_cough_voice, sr = librosa.load('/home/madhu/work/cough_classification/data/sorted_recording/Wu0376/voices/voice_3.wav')





draw_audio_spec(audio_data_cough_mc, sr, 'Cough from main mic')

draw_audio_spec(audio_data_cough_st_1, sr, 'Cough from stethoscope 1')

draw_audio_spec(audio_data_cough_st_2, sr, 'Cough from stethoscope 2')







## This is the experiment about loud and quiet sounds 
audio_data_low, sr = librosa.load('/home/madhu/work/cough_classification/data/sorted_recording/Wu0442/coughs_mc/cough_10.wav')

audio_data_high, sr = librosa.load('/home/madhu/work/cough_classification/data/sorted_recording/Wu0376/coughs_mc/cough_10.wav')


draw_audio_spec(audio_data_low, sr, 'Quiet Cough')

draw_audio_spec(audio_data_high, sr, 'Loud Cough')


audio_data_low = audio_data_low*3

plt.figure()
plt.plot(audio_data_low)







## Just generate some random positve numbers for representing the acceleration magnitudes
import random
import matplotlib.pyplot as plt
import numpy as np

N = 500
sr = 100
acc_values = []

acc_values.extend([0]*50)

for i in range(N):
    acc_values.append(random.random())

acc_values.extend([0]*50)

fig, ax = plt.subplots(figsize=(20, 10))
x_axis = np.linspace(0, len(acc_values)/sr, len(acc_values), endpoint=False)
# x_axis = range( len(acc_values) )
# plt.title('Accelerometer Magnitude', fontsize=25)
plt.plot(x_axis, acc_values)
plt.xlabel('Time (S)', fontsize=20)
plt.ylabel('Accelerometer Magnitude', fontsize=20)
plt.show()








