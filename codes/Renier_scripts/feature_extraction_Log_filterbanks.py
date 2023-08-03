
import os
import sys
import csv
import math
import pandas as pd
import numpy as np
import shutil

from glob import glob
from natsort import natsorted

from scipy.fftpack import fft, fftfreq
from scipy.io.wavfile import read
from scipy.signal import hamming
import librosa
from log_filterbanks import log_filterbanks as log_fbanks

import matplotlib.pyplot as plt

import warnings

warnings.simplefilter(action = "ignore",category = FutureWarning)
warnings.simplefilter(action = "ignore",category = UserWarning)

coughs_dir = "/home/madhu/work/cough_classification/Renier_data/Code/Classifiers/ANN/data/coughs_single/"
base_dir = "/home/madhu/work/cough_classification/Renier_data/Code/Classifiers/ANN/data/Features_Logfbanks/"
features_dir = base_dir + "features_windows/"
labels_csv = "/home/madhu/work/cough_classification/Renier_data/Code/Classifiers/ANN/data/PNames_and_Labels_no_pipe.csv"

# framerate
fs = 44100

# Window length of 25ms = 1100 samples
h = 1100
# Frameskip of 10ms = 440 samples
r = h/4

N_STACKS = 1
WIN_SKIP = 1

# Extract features from coughs?
EXTRACT = True
# Copy files to all folder?
COPY_ALL = True
# Convert extracted files to supervectors?
CONVERT_SUPERVECTORS = True 
# Create one csv with all data?
MAKE_COMPLETE_DATASET = True


def get_colNames(num_fbanks):
	"""
	Just doing Log of FFT filterbanks
	"""

	col_Names = []
	for k in range(num_fbanks):
		col_Names.append("Log_fbank_"+str(k+1))

	return col_Names

def get_StudyNum(f):
	parts = os.path.basename(f).split("_")

	return parts[0] + "_" + parts[1]

def get_label(rec_name):
	patient_names = meta_data[:,0]
	labels = meta_data[:,1]
	for p in patient_names:
		if p in rec_name:
			idx = list(patient_names).index(p)
			return float(labels[idx])
		else:
			continue

def extract_features(cough_wavs, dir, meta_data):

	"""
	Perform feature extraction on all cough recordings
	in cough_wavs
	"""
    
	# Setup the output directories
	rec_name = os.path.basename(dir)
	out_dir = features_dir + rec_name
	if(os.path.isdir(out_dir) != True):
		os.mkdir(out_dir)
	
    # feature_vecs = []
	for wav in cough_wavs:
        
		fs,audio = read(wav)
        # audio, fs = librosa.load(wav)
        
		log_filterbanks = log_fbanks(audio, nfilters=80, spec='pow')
        
		nframes, nfilters = log_filterbanks.shape
        
		print(log_filterbanks.shape)
        
		colNames = get_colNames(nfilters)
        
		# Write the extracted features to disk
		wav_name = os.path.basename(wav).split(".")[0]
        
		f_out_name = features_dir + rec_name + "/" + wav_name + ".csv"
		
		df = pd.DataFrame(log_filterbanks,columns = colNames)
		
		df["TBResult"] = get_label(rec_name)
        
		df.to_csv(f_out_name, sep = ';',index = False)



def copy_to_all():

	all_dir = features_dir + "all/"
	if os.path.isdir(all_dir) != True:
		os.mkdir (all_dir)

	dirs_list = glob(features_dir + "C*")
	for dir in dirs_list:
		files = glob(dir + "/*.csv")

		for f in files:
			shutil.copy(f,all_dir)

def get_SV_cols(cols):
	"""

	Create new list of columns that is
	N_STACKS  x len(cols) in the following format:

	Eg, if N_STACKS = 2 and cols = [col1, col2, col3]:

	SV_cols = [col1_1, col1_2, col2_1, col2_2, col3_1, col3_2]

	"""
	SV_cols = []

	for col in cols:

		for k in range(1,N_STACKS+1):

			colname = col + "_" + str(k)

			SV_cols.append(colname)

	SV_cols.append("TBResult")


	return SV_cols

def create_SV_dataset(f):

	data = pd.read_csv(f,sep=';')

	if N_STACKS > 1:
		columns = list(data)[:-1]
		SV_cols = get_SV_cols(columns)
	else:
		SV_cols = list(data)

	# convert data to array (easier for stacking)
	data = np.array(data)
	# get label
	label = data[0,-1]
	# now remove the label data
	data = data[:,:-1]

	# Number of super vectors to create from f
	num_vecs = (len(data) - N_STACKS) // WIN_SKIP

	SV_df = []
	# for each row
	for k in range(num_vecs+1):
		SV_row = []
		# for each column
		for i in range(data.shape[1]):

			if N_STACKS > 1:
				SV_row.append( [data[k:k+N_STACKS,i][j] for j in range(N_STACKS)])

			elif N_STACKS == 1:
				SV_row.append([data[k,i]])

		SV_row = [item for sublist in SV_row for item in sublist]
		SV_row.append(label)
		SV_df.append(SV_row)

	SV_df = np.vstack(SV_df)
	SV_df = pd.DataFrame(SV_df, columns = SV_cols )

	return SV_df

def convert_to_super_vector():

	"""
	This function script reads in the feature data from
	features_dir and creates a super vector that is 
	N_STACKS x n_columns long.


	Psuedo Code:

	for each file in features_dir:
		read in data
		create N_STACKS duplicates of each column

		for k in range(0 -> (len(data)-WIN_SKIP)//N_STACKS):
			append each stack horizontally 

		save to vectors_dir
	"""

	SV_base_dir =  base_dir + "super_vectors/windows/"
	if os.path.isdir(SV_base_dir) != True:
		os.mkdir(SV_base_dir)

	SV_dir = SV_base_dir + str(N_STACKS) + "/"
	# Check that the output dirrectory exists
	if os.path.isdir(SV_dir) != True:
		os.mkdir(SV_dir)


	features_flist = glob(features_dir + "all/*.csv")

	# For all the .csv feature files:
	for f in features_flist:

		# Create a dataset with N_STACKS supervectors
		print ("Running on",os.path.basename(f))
		SV_df = create_SV_dataset(f)

		# Save the new dataset to disk
		f_out = SV_dir + os.path.basename(f)
		# print "Saving to",f_out
		SV_df.to_csv(f_out,sep=';',index=False,index_label=False)

def make_complete_dataset(dir):

	"""

	Load all the cough feature data 
	in the supervector directory
	into one DataFrame

	Also add a column with the patient name

	"""
	
	f_full_dataset = base_dir + "complete_windows/full_dataset_windows_"+str(N_STACKS)+"_stacks.csv"

	features_list = glob(dir + "*.csv")

	# first get the columns
	col_Names = list(pd.read_csv(features_list[0],sep=';',nrows=0))


	dataset = pd.DataFrame(columns = col_Names)

	for f in features_list:
		StudyNum = get_StudyNum(f)
		data = pd.read_csv(f,sep=';')

		if list(data) != col_Names:
			print (f)
			print (col_Names)
			print (data)
			exit()

		data["StudyNum"] = StudyNum
		dataset = dataset.append(data,ignore_index = True)


	print ("saving full dataset to ",f_full_dataset)
	dataset.to_csv(f_full_dataset,index=False,index_label=False,sep=';')


	print ("Full dataset shape:")
	print (dataset.shape)
	return dataset


def exit():
	sys.exit(0)



if __name__ == '__main__':

	meta_data = np.loadtxt(labels_csv,delimiter=";",dtype = str,skiprows = 1)

	coughs_dir_list = glob(coughs_dir + "C_364*")

	recordings_list = [os.path.basename(f) for f in coughs_dir_list ]


	if EXTRACT:
		for dir in coughs_dir_list:
			print ("Extracting features for coughs in: ", dir)
			# List of cough wav files sorted numerically
			cough_wavs = natsorted(list(glob(dir + "/*.wav")))
			extract_features(cough_wavs, dir, meta_data)
			print ("done\n")

	# Copy all generated feature files to
	# the all folder
	if COPY_ALL:
		print ("Copying features to all folder...")
		copy_to_all()
		print ("done!\n")

		print ("\ndone\n")


	if CONVERT_SUPERVECTORS:
		print ("Converting data into supervectors...")
		convert_to_super_vector()
		print ("done\n")


	if MAKE_COMPLETE_DATASET:
		dir = base_dir + "super_vectors/windows/" + str(N_STACKS) + "/"
		print ("Making one csv from all files in",dir)
		make_complete_dataset(dir)
		print ("done\n")
		



		
