import os
import sys
import numpy as np
import pandas as pd

from scipy.fftpack import fft, fftfreq
from scipy.io.wavfile import read, write
from scipy.signal import hamming

import librosa

import matplotlib.pyplot as plt

"""
Used and modified this :
http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/


Got macspec and powspec stuff from sigproc libraries
"""

PLOT = False
PLOT_FILTERS = False

def get_filterbanks(nfilters=10,fmin=0,fmax=None,fs=44100,nfft=512):
    # starting frequencies of each filter
    freqs = np.linspace(fmin, fmax, nfilters+2)
    # put starting frequencies into same bins of fft
    bin = np.floor((nfft+1)*(freqs/fs))

#   coeff_mat = np.zeros( (nfilters, nfft/2 + 1))
    coeff_mat = np.zeros( (nfilters, (int(nfft/2) + 1)))
    for i in range(nfilters):
        # left slope
        for j in range( int(bin[i]), int(bin[i+1]) ):
            coeff_mat[i,j] = (j - bin[i]) / (bin[i+1] - bin[i])

        # right slope
        for j in range( int(bin[i+1]), int(bin[i+2])):
            coeff_mat[i,j] = (bin[i+2] - j) / (bin[i+2]-bin[i+1])

    # Plot the triangular filters
    if PLOT_FILTERS:
        temp_axis = np.linspace(fmin,fmax,nfft/2+1)
        for k in range(nfilters):
            plt.plot(temp_axis,coeff_mat[k,:])
            plt.show()

        exit()

    return coeff_mat

def magspec(frames,NFFT):
	"""
	Compute the magnitude spectrum of each
	frame in frames

	INPUT:
	========
	frames:	each row is a frame
	NFFT:	FFT length. If NFFT>frame_len, frames are zero padded
	
	OUTPUT:
	========
	[n_frames x NFFT] shaped magnitude spectrum	
	"""

	spectrum = np.fft.rfft(frames, n = NFFT)
	return np.abs(spectrum)

def powspec(frames, NFFT):
	"""
	Compute the magnitude spectrum of each
	frame in frames

	INPUT:
	========
	frames:	each row is a frame
	NFFT:	FFT length. If NFFT>frame_len, frames are zero padded
	
	OUTPUT:
	========
	[n_frames x NFFT] shaped power spectrum	
	"""
	ms = magspec(frames, NFFT)
	return 1.0/NFFT * np.square(ms)

def log_powspec(frames, NFFT, norm=False):
	ps = powspec(frames,NFFT)
	# fix small values
	ps[ps<=1e-30] = 1e-30

	log_powspec = 10*np.log10(ps)

	if norm:
		return log_powspec - np.max(log_powspec)
	else:
		return log_powspec


def apply_fbanks(audio, fs=44100, nfilters=10, winlen=512, winstep=256, nfft=512, fmin=0, fmax=None, winfunc=hamming, spec=None):
	"""
	Apply linear filterbanks to magnitude and
	power spectrum of audio
	"""
	fmax = fmax or fs/2

	# Break up audio into frames
	"""
	Changed to using librosa for framing process
	"""
	frames = librosa.util.frame(audio, frame_length=winlen, hop_length=winstep)

	# Apply hamming window to each frame
	frames = np.array([frames[:,k]*hamming(winlen) for k in range(frames.shape[1])])

	# Get filterbank coefficients
	fbanks = get_filterbanks(nfilters,fmin,fmax,fs,nfft)

	# Some Plotting stuff
	if PLOT:

		"""
		if plotting flag is set, need to work out both
		"""
		pspec = powspec(frames, nfft)
		# apply filters to power spectrum
		pspec_fbanks = np.dot(pspec,fbanks.T)
		pspec_fbanks = np.where(pspec_fbanks==0,np.finfo(float).eps, pspec_fbanks)

		mspec = magspec(frames, nfft)
		# apply filters to magnitude spectrum
		magspec_fbanks = np.dot(mspec,fbanks.T)
		magspec_fbanks = np.where(magspec_fbanks==0,np.finfo(float).eps, magspec_fbanks)

		# mag and pow_filtered is a 10x25x257 matrix
		# -> 10 filters applied over 25 frames of length 257 each.
		# **assuming default conditions
		mag_filtered = []
		pow_filtered = []
		for k in range(nfilters):
			mag_filtered.append(np.multiply(mspec,fbanks[k]))
			pow_filtered.append(np.multiply(pspec,fbanks[k]))

		mag_filtered = np.asarray(mag_filtered)
		pow_filtered = np.asarray(pow_filtered)

		freqs = np.linspace(0,fmax,nfft/2+1)
		
		# Filter number to use in plotting
		plt_filter_num = 1

		# plot for each frame
		for k in range(np.shape(pspec)[0])[:5]:
			plt.plot(audio)
			plt.subplot(3,2,1)
			plt.title("Audio (Framed and windowed)")
			# plt.xlabel("Time")
			plt.plot(frames[k])
			plt.subplot(3,2,3)
			plt.title("Magnitude Spectrum")
			# plt.xlabel("Frequency")
			plt.plot(freqs,mspec[k])
			plt.subplot(3,2,5)
			plt.title("Power Spectrum")
			plt.plot(freqs, pspec[k])

			# now windows by filterbank 2
			plt.subplot(3,2,2)
			plt.title("Filter 2")
			plt.plot(fbanks[plt_filter_num-1])

			plt.subplot(3,2,4)
			plt.title("Mag Spectrum Filtered")
			# plt.ylim(max(magspec[k]))
			plt.plot(freqs,mag_filtered[plt_filter_num-1,k,:])
			
			plt.subplot(3,2,6)
			plt.title("Power Spectrum Filtered")
			# plt.ylim(max(pspec[k]))
			plt.plot(freqs,pow_filtered[plt_filter_num-1,k,:])
			
			plt.show()

	if spec == 'pow':
		pspec = powspec(frames, nfft)

		"""
		Important:
		Dot product already does summation so no need 
		to do sum later...	
		"""
		# apply filters to power spectrum
		pspec_fbanks = np.dot(pspec,fbanks.T)
		pspec_fbanks = np.where(pspec_fbanks==0,np.finfo(float).eps, pspec_fbanks)

		return pspec_fbanks

	elif spec == 'mag':
		mspec = magspec(frames, nfft)
		# apply filters to magnitude spectrum
		magspec_fbanks = np.dot(mspec,fbanks.T)
		magspec_fbanks = np.where(magspec_fbanks==0,np.finfo(float).eps, magspec_fbanks)
		
		return magspec_fbanks

	else:
		print ("ERROR: spec != 'pow' or 'mag'")
		exit()

def log_filterbanks(audio, nfilters=10, fs=44100, winlen=512, winstep=256, nfft=512, spec=None):
	"""
	Calculate the log filterbank values for audio.
	This breaks up signal into windows with winlen and winstep
	Then applies nfilters number of linear filterbanks to the
	magnitude or powerspectrum 
	"""
	spec = spec or "pow"
	assert (spec == "mag" or spec == "pow"), ("Unknown spectrum choice %s"%spec)

	fbanks = apply_fbanks(audio 	= audio,
						  fs 		= fs,
						  winlen 	= winlen,
						  winstep 	= winstep,
						  nfilters 	= nfilters,
						  nfft 		= nfft,
						  spec 		= spec)
	# Take log
	log_fbanks = np.log(fbanks)

	return log_fbanks

def test():
	wav = "../data/coughs/C_407/C_407_cough_0.wav"
	fs,audio = read(wav)

	NFFT = 2**10
	log_fbanks = log_filterbanks(audio,
								 nfilters	= 10,
								 fs 		= 44100,
								 winlen 	= 512,
								 winstep 	= 512,
								 nfft 		= 2048,
								 spec 		= "pow")

	print ('n_fft = ',NFFT)
	print (log_fbanks.shape)


if __name__ == '__main__':
	
	test()

	



