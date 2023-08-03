


import os
import librosa
import numpy as np
import pandas as pd

from collections import defaultdict

from glob import glob
from natsort import natsorted
from scipy.io.wavfile import read
from scipy.stats import kurtosis
from scipy.signal import hamming

# from progress_bar import printProgress
from log_filterbanks import log_filterbanks
from helper import *
from feature_extraction_MFCC import calc_mfcc, get_mfcc_means, calc_MFCC_D_A

import matplotlib.pyplot as plt


fs = 44100
OVERLAP = False


def get_binned_framenums(x, n_frames, conf_dict):
	"""
	Get the indices of which frames 
	fit into which bins.

	"""
	empty_bin_flag = False

	N = conf_dict['N']
	M = conf_dict['M']
	B = conf_dict['B']

	n_samples = len(x)

	if B > 1:

		if B <= n_frames:
			# Size of each bin in samples
			bin_size = n_samples / B

			# edges of bins in samples
			sample_edges = [n_samples/B * k for k in range(1,B+1)]

			# print len(x),sample_edges
			bins = []
			drop = defaultdict(list)
			# which frame to start at
			i = 0
			# which bin we are adding to
			bin_no_count = 1
			for s in sample_edges:
				# print s,i,n_frames
				# the frames that fit into this bin
				b = []
				for k in range(i,n_frames):
					frame_start = k * M
					frame_end = frame_start + N
					# print 'f_s=\t',frame_start,'\tf_e=',frame_end,'\tbin_edge=',s,
					# frame fits completely
					if frame_start < s and frame_end <= s:
						# print 'complete:'
						b.append(k)
						i += 1
						continue

					# frame is overlapping or exactly between two bins.
					elif frame_start < s and frame_end > s:
						# leaning towards this bin or exactly between bins
						if abs(frame_start-s) >= abs(frame_end-s):
							# print 'overlap this bin'
							# print abs(frame_start-s),abs(frame_end-s)
							# drop[bin_no_count].append(k)
							b.append(k)
							i += 1
							continue

						# leaning towards next bin
						else:
							# print 'overlap next bin'
							continue
					# frame is not in this bin
					else:
						# print 'skip'
						continue


				bins.append(b)
				bin_no_count += 1


			"""
			Now check if we should keep or drop
			overlapping frames
			"""
			drop = dict(drop)
			for key, val in drop.items():
				# we can drop the overlapping frames
				if bins[key-1]:
					continue
				# if there are no frames in this bin
				# keep all overlapping frames that would
				# have gone into that bin
				else:
					bins[key-1] = val
				


			"""
			Lastly, check if the bins still
			has one bin thats empty
			"""
			for b in bins:
				if b:
					continue
				# bin is empty
				else:
					empty_bin_flag = True
					break


		# If there are less frames than bins
		# fill each bin with all the frames
		else:
			bins = [range(0,n_frames) for k in range(B)]

		bins = np.array(bins)

	else:
		bins = np.array(np.arange(n_frames))

	return bins, empty_bin_flag


def average_over_bins(df, full_df = False):
	"""
	Average the features over all frames in each bin


	full_df:	If true, perform averaging on
				an entire df, and not one containing one 
				recording
	"""
	num_bins = len(np.unique(df.Bin_No.values))

	# Won't need this if we are averaging over windows
	df = df.copy().drop('Win_No',axis=1)

	# Convert numeric columns to the appropriate dtypes
	df = df.apply(lambda x: pd.to_numeric(x, errors = 'ignore'))
	# Small df verion
	if not full_df:
		means_list = []
		for k in range(1,num_bins+1):
			temp_df = df[df.Bin_No == k]
			means = list(temp_df.mean(numeric_only = True).values)
			means.insert(0,list(df.StudyNum.values)[0])
			means_list.append(means)

		means_list = np.vstack(means_list)
		avg_df = pd.DataFrame(means_list,columns = list(df))

	# Do averaging for each cough in large dataset
	else:
		avg_df = pd.DataFrame(columns = list(df))
		# For each recording
		for name, group in df.groupby("StudyNum"):
			# For each cough in each recording
			temp_df2 = pd.DataFrame(columns=list(df))
			for n_cough, cough_df in group.groupby('Cough_No'):
				means_list = []
				# For each bin in each cough
				for b, bin_df in cough_df.groupby("Bin_No"):
					means = list(bin_df.mean(numeric_only=True).values)
					means.insert(0,name)
					means_list.append(means)

				means_list = np.vstack(means_list)
				temp_df = pd.DataFrame(means_list,columns = list(df))

				temp_df2 = temp_df2.append(temp_df)


			avg_df = avg_df.append(temp_df2)

	return avg_df


def extract_features_Filterbank(dir, coughs_list, conf_dict):

	rec = os.path.basename(dir)

	if 'CONX' in rec:
		TBResult = 0
	else:
		TBResult = 1

	N = conf_dict['N']
	M = conf_dict['M']
	B = conf_dict['B']
    
    # Madhu: testing
    N = 2048
    M = 2048
    B = 1
    
    feat_colNames = ["StudyNum",'Bin_No','Cough_No','Win_No',"ZCR","Kurtosis","LogE"]
    N_FBANKS = 140
    for k in range(1,N_FBANKS+1):
        feat_colNames.append(''.join(['FBANK_',str(k)]))
   	feat_colNames.append('TBResult')
    
    # Initialize full dataframe
    feat_colNames = get_colNames(N_FBANKS = N_FBANKS)

    df = pd.DataFrame(columns = feat_colNames)

    # for each cough in this recording
    for wav in coughs_list:
        # Get number of this cough
        cough_no = wav.split("_")[-1].split(".")[0]

        # load in audio
        _,x = read(wav)

        # frame audio
        frames = librosa.util.frame(x, frame_length = N, hop_length = M)

        """
        Get which frames go into which bins
        """
        bins,empty_bin_flag = get_binned_framenums(x, frames.shape[1], conf_dict)


        """
        If one of the bins are empty, just leave out this frame
        """
        if empty_bin_flag:
            # print 'skipping',os.path.basename(wav)
            # print to disk which wavs this happen to
            f_temp = "".join(['../temp/bins_test/WAVS_LEFT_OUT.N_FBANKS=',str(N_FBANKS),'.N=',str(N),'.B=',str(B),'.txt'])
            with open(f_temp,'a') as f_temp_out:
                line = os.path.basename(wav)+'\tn_wavs_in_rec='+str(len(coughs_list))+'\n'
                f_temp_out.write(line)

            continue

        """
        Calculate the Filterbanks
        This is the frequency spectrum binned into
        N_FBANKS number of 'filterbanks', which are
        triangular filters linearly spaced on the 
        frequency axis
        """
        FBANKS = log_filterbanks(x,
                                nfilters 	= N_FBANKS,
                                fs 		= fs,
                                winlen 	= N,
                                winstep 	= M,
                                nfft 		= N,
                                spec 		= 'pow')
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

			vec = [rec,cough_no,k,kurt,zcr,logE]

			# Add the Filterbanks
			vec.extend(FBANKS[k,:])
			# Add meta_data
			vec.append(TBResult)

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
		temp_df = pd.DataFrame(feature_matrix, columns = feat_colNames)

		"""
		We need to duplicate the temp_df for each bin
		"""
		if B > frames.shape[1]:
			for b in range(1,B+1):
				temp_df['Bin_No'] = b
				df = df.append(temp_df)

		else:
			df = df.append(temp_df)


	# Convert numeric columns to the appropriate dtypes
	df = df.apply(lambda x: pd.to_numeric(x, errors = 'ignore'))
	return df



def extract_features_MFCC(dir, coughs_list, conf_dict):

	rec = os.path.basename(dir)

	if 'CONX' in rec:
		TBResult = 0
	else:
		TBResult = 1

	N = conf_dict['N']
	M = conf_dict['M']
	B = conf_dict['B']
    
    # Madhu: testing
    N = 2048
	M = 2048
	B = 1
    
    feat_colNames = ["StudyNum",'Bin_No','Cough_No','Win_No',"ZCR","Kurtosis","LogE"]
    # N_FBANKS = 140
    for k in range(N_MFCC):
            feat_colNames.append("MFCC" + str(k))
    for k in range(N_MFCC):
            feat_colNames.append("MFCC_D" + str(k))
    for k in range(N_MFCC):
            feat_colNames.append("MFCC_A" + str(k))
    # for k in range(1,N_FBANKS+1):
    #   feat_colNames.append(''.join(['FBANK_',str(k)]))
    feat_colNames.append('TBResult')
    
    # Initialize full dataframe
#   feat_colNames = get_colNames(N_FBANKS = N_FBANKS)
    
	df = pd.DataFrame(columns = feat_colNames)
    
    mfcc_means = get_mfcc_means(coughs_list, N, M, N_MFCC) 
    
	# for each cough in this recording
	for wav in coughs_list:
		# Get number of this cough
		cough_no = wav.split("_")[-1].split(".")[0]

		# load in audio
		_,x = read(wav)
        
        # x, fs = librosa.load(wav)
        
		# frame audio
		frames = librosa.util.frame(x, frame_length = N, hop_length = M)

		"""
		Get which frames go into which bins
		"""
		bins, empty_bin_flag = get_binned_framenums(x, frames.shape[1], conf_dict)


		"""
		If one of the bins are empty, just leave out this frame
		"""
		if empty_bin_flag:
			# print 'skipping',os.path.basename(wav)
			# print to disk which wavs this happen to
			f_temp = "".join(['../temp/bins_test/WAVS_LEFT_OUT.N_FBANKS=',str(N_FBANKS),'.N=',str(N),'.B=',str(B),'.txt'])
			with open(f_temp,'a') as f_temp_out:
				line = os.path.basename(wav)+'\tn_wavs_in_rec='+str(len(coughs_list))+'\n'
				f_temp_out.write(line)

			continue

		"""
		Calculate the Filterbanks
		This is the frequency spectrum binned into
		N_FBANKS number of 'filterbanks', which are
		triangular filters linearly spaced on the 
		frequency axis
		"""
		FBANKS = log_filterbanks(x,
								 nfilters 	= N_FBANKS,
								 fs 		= fs,
								 winlen 	= N,
								 winstep 	= M,
								 nfft 		= N,
								 spec 		= 'pow')
        
        # MFCC_vec = calc_MFCC_D_A(np.asarray(x), mfcc_means, N, M, N_MFCC)
        # MFCC_vec = calc_MFCC_D_A(x.astype(np.float32), mfcc_means, N, M, N_MFCC)
        MFCC, MFCC_D, MFCC_A = calc_MFCC_D_A(x.astype(np.float32), mfcc_means, N, M, N_MFCC)
        
        MFCC_vec = np.vstack((MFCC, MFCC_D, MFCC_A)).transpose()
        
        
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

            vec = [rec,cough_no,k,kurt,zcr,logE]
            vec.extend(MFCC_vec[k,:])
            # Add meta_data
            vec.append(TBResult)

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
        temp_df = pd.DataFrame(feature_matrix, columns = feat_colNames)

        """
        We need to duplicate the temp_df for each bin
        """
        if B > frames.shape[1]:
            for b in range(1,B+1):
                temp_df['Bin_No'] = b
                df = df.append(temp_df)

        else:
            df = df.append(temp_df)


    # Convert numeric columns to the appropriate dtypes
    df = df.apply(lambda x: pd.to_numeric(x, errors = 'ignore'))
    return df








def calc_feat_df(coughs_dir_list,conf_dict):

# 	feat_colNames = get_colNames(N_FBANKS = N_FBANKS)
    df = pd.DataFrame(columns = feat_colNames)

	k = 0.0
	for dir in coughs_dir_list:
        print(dir)
		coughs_list = natsorted(glob(dir + "/*.wav"))

		feat_matrix = extract_features(dir, coughs_list, conf_dict)

		print (feat_matrix)
		exit()

		temp_df = pd.DataFrame(feat_matrix, columns = feat_colNames)

		df = df.append(temp_df)

		k += 1

		printProgress(k, len(coughs_dir_list),prefix='Progress:',suffix='Complete',barLength=50)

	return df


def make_dict(f):

    import re

    f_ = os.path.basename(f)
    parts = f_.split('.')

    if 'FBANKS' in f_:
        f_num = re.findall('N_FBANKS=(\d+)',f_)[0]

    else:
        f_num = re.findall('N_MFCC=(\d+)',f_)[0]


    d = {'Feature' : f_num,
         'N' : 	parts[1].split('=')[1],
         'M': 	parts[2].split('=')[1],
         'B': 	parts[3].split('=')[1],
         'Avg': parts[4].split('=')[1]}

    return d


if __name__ == '__main__':

	"""
	1.	Select specific feature extraction parameters (best results)
	2.	Do feature extraction:
		For R = [0,10,20,30]:
			reduce signal with R% from left and right
			save features in phase_testing/R%/


	"""
    
    N_FBANKS = 100
    
	path = '/home/madhu/work/cough_classification/Renier_data/Code/Experiments/insights/'

	# DEFINE INPUT / OUTPUT DIRS
	coughs_dir = '/home/madhu/work/cough_classification/Renier_data/Code/Experiments/param_optimization/data/coughs/'
	coughs_dir_list = natsorted(glob(coughs_dir+'C*'))

	features_list = glob(os.path.join(path,'data/features/best_reduced_LR/*'))

	for R in [0,10,20,30]:

		features_output_dir = os.path.join(path,'data/features/phase_testing/{}%/'.format(R))

		for f in features_list:

			d = make_dict(f)

			# Setup output filename
			f_out = os.path.join(features_output_dir, os.path.basename(f))

			print ("FEATURE EXTRACTION WITH PARAMS:",d)
			df = calc_feat_df(coughs_dir_list,d)
			print ('done\n')


			if df.isnull().any().any():
				print ("\n\nNAN Value in DF:")
				print (f_out)
				exit()

			else:
				"""
				Write the full dataset to disk
				"""
				# print "Writing data to",os.path.basename(f_out)
				df.to_csv(f_out,sep=';',index=False,index_label=False)


