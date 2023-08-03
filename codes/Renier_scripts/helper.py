"""
A helper script
"""

import numpy as np
import pandas as pd
import os
from collections import defaultdict

def conv_df_values(df):
	# Convert numeric columns to the appropriate dtypes
	df = df.apply(lambda x: pd.to_numeric(x, errors = 'ignore'))

	return df

def get_colNames(d):

	col_Names = ["StudyNum",'Bin_No','Cough_No','Win_No',"ZCR","Kurtosis","LogE"]

	if 'N_MFCC' in d:
		N_MFCC = int(d['N_MFCC'])

		for k in range(0,N_MFCC):
			col_Names.append("MFCC" + str(k))

		# MFCC_D
		for k in range(0,N_MFCC):
			col_Names.append("MFCC_D" + str(k))

		# MFCC_A
		for k in range(0,N_MFCC):
			col_Names.append("MFCC_A" + str(k))

	elif 'N_FBANKS' in d:
		N_FBANKS = int(d['N_FBANKS'])
		for k in range(1,N_FBANKS+1):
			col_Names.append(''.join(['FBANK_',str(k)]))

	else:
		print ("ERROR in:\nget_colNames(N_MFCC = None, MFCC_D = True, MFCC_A = True, N_FBANKS = None)")
		print ("N_MFCC and N_FBANKS both = None")
		exit()


	col_Names.append('TBResult')


	return col_Names

def zero_crossing_rate(wavedata):

    zero_crossings = 0
    
    number_of_samples = len(wavedata)
    for i in range(1, number_of_samples):
        
        if ( wavedata[i - 1] <  0 and wavedata[i] >  0 ) or \
           ( wavedata[i - 1] >  0 and wavedata[i] <  0 ) or \
           ( wavedata[i - 1] != 0 and wavedata[i] == 0):
                
                zero_crossings += 1
                
    zero_crossing_rate = zero_crossings / float(number_of_samples-1)

    return zero_crossing_rate

def LogEnergy(audio):

	eps = 0.0001
	N = len(audio)
	energy = abs(audio**2)
	LogE =10 * np.log10(eps + sum(energy) / float(N))

	return LogE

def load_conf_dict(cfg):
	d= {}
	with open(cfg) as f:
			for line in f:
				(key,val) = line.rstrip().split('=')
				try:
					d[key] = int(val)
				except ValueError:
					d[key] = val
	return d

def make_fname_from_dict(d, dir, N_MFCC = None, N_FBANKS = None, spec = None, extract = False):

	"""
	d:		dictionary with config params
	dir:	directory of features

	"""

	if N_MFCC != None:

		if extract:

			MFCC_OUT_DIR = dir + '/N_MFCC=' + str(N_MFCC)

			# If the output dir doesnt exist
			if not os.path.isdir(MFCC_OUT_DIR):
				os.mkdir(MFCC_OUT_DIR)
			# Setup output filename
			f_out = "".join([MFCC_OUT_DIR,
						 	'/features.N=',str(d['N']),
						 	'.M=',str(d['M']),
						 	'.B=',str(d['B']),
						 	'.Avg=',d['Avg'],
						 	'.N_MFCC=',str(N_MFCC),
						 	'.csv'
						 	])

			return f_out

		else:
			"""
			Just making a filename
			"""

			if str(dir).endswith('/'):
				dir = str(dir)[:-1]
				

			f_out = "".join([dir,
						 	'/features.N=',str(d['N']),
						 	'.M=',str(d['M']),
						 	'.B=',str(d['B']),
						 	'.Avg=',d['Avg'],
						 	'.N_MFCC=',str(N_MFCC),
						 	'.csv'
						 	])

			return f_out


	elif N_FBANKS != None:

		# Make spec type default to magnitude spectrum
		if spec == None:
			spec = 'mag'

		
		if extract:
			"""
			This is when we want to create the directory 
			for feature extraction

			- Making a directory from the config params
			"""
			FBANKS_OUT_DIR = dir + '/N_FBANKS=' + str(N_FBANKS) + '_SPEC=' + spec

			if not os.path.isdir(FBANKS_OUT_DIR):
				os.mkdir(FBANKS_OUT_DIR)

			f_out = "".join([FBANKS_OUT_DIR,
						 	'/features.N=',str(d['N']),
						 	'.M=',str(d['M']),
						 	'.B=',str(d['B']),
						 	'.Avg=',d['Avg'],
						 	'.N_FBANKS=',str(N_FBANKS),
						 	'.csv'
						 	])
		
			# check if the file already exists
			return f_out

			# else:
				# print '\n\n WARNING!!\nFeature file already exists, are you sure you want to overwrite?'
				# print 'dict = ',d
				# print 'dir 	= ',dir
				# print 'f_out = ',f_out
				# choice = raw_input('\n[Y/N]')
				# if choice == 'N':
				# 	exit()


		else:
			"""
			Just making a filename using the dir
			"""
			f_out = "".join([dir,
						 	'features.N=',str(d['N']),
						 	'.M=',str(d['M']),
						 	'.B=',str(d['B']),
						 	'.Avg=',d['Avg'],
						 	'.N_FBANKS=',str(N_FBANKS),
						 	'.csv'
						 	])

			return f_out

	else:
		print ("ERROR in:\nmake_fname_from_dict(d, dir, N_MFCC = None, N_FBANKS = None, spec = None, create_dir = False):")
		print ("N_MFCC and N_FBANKS both = None")
		exit()

	
def get_binned_framenums(x, n_frames, B):
	"""
	Get the indices of which frames 
	fit into which bins.

	"""
	empty_bin_flag = False

	N = n_frames
	M = n_frames

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
		for name, group in df.groupby("Study_Num"):
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



