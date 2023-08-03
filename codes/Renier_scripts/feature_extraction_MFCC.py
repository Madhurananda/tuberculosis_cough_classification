import numpy as np
import librosa
from scipy.signal import hamming
from scipy.io.wavfile import read


# fs = 44100
# OVERLAP = False

def calc_mfcc(x, N, M = None, window = hamming, N_MFCC=13, fs=44100):
    
	"""
	Calculate mfccs from x
    
	N: window length
	M: frameskip
    
	by default, N_FFT = N
    
	win_length and hop_length given in samples
	"""
	
	N_FFT = N
    
	# Compute power spectrum
	D = np.abs(librosa.stft(x,
							win_length 	= N,
							hop_length 	= M,
							n_fft 		= N_FFT,
							window 		= window,
							center 		= False))**2
	
    # Compute mel spectrogram
	S = librosa.feature.melspectrogram(S = D, sr = fs, htk = True)
    
	# Get MFCCs from log mel spectrogram
	MFCC = librosa.feature.mfcc(S = librosa.core.amplitude_to_db(S), n_mfcc = N_MFCC, sr = fs)
    
	return MFCC


# def get_mfcc_means(all_cough_audio, N, M, N_MFCC, fs):
def get_mfcc_means(all_cough_audio, N, M, N_MFCC):

    """
    Calculate the means of the N_MFCC MFCCs
    over all the coughs in cough_wavs
    """
    
    
    # Calculate MFCCs with M = N = N_FFT
    # mfccs = calc_mfcc(x = np.asarray(all_cough_audio), N = N, M=M, N_MFCC=N_MFCC, fs=fs)
    mfccs = calc_mfcc(x = np.asarray(all_cough_audio), N = N, M=M, N_MFCC=N_MFCC)

    means = list(np.mean(mfccs,axis=1))

    return means


# def calc_MFCC_D_A(audio, mfcc_means, N, M, N_MFCC, fs):
def calc_MFCC_D_A(audio, mfcc_means, N, M, N_MFCC):

    # MFCC = calc_mfcc(audio, N = N, M = M, N_MFCC=N_MFCC, fs=fs)
    MFCC = calc_mfcc(audio, N = N, M = M, N_MFCC=N_MFCC)


    # Subtract cepstral means
    mfcc_means = np.asarray(mfcc_means)
    MFCC = np.transpose(MFCC.T - mfcc_means)
    
    # print('MFCC shape, ', MFCC.shape)
    
    if MFCC.shape[1] % 2 == 0:
        w = MFCC.shape[1] - 1
    else:
        w = MFCC.shape[1]
    
    if w<3:
        print('MFCC shape is: ', MFCC.shape, ' and thus no further features have been extracted for cough:', audio)
        MFCC_D = []
        MFCC_A = []
    else:
        MFCC_D = librosa.feature.delta(MFCC, width=w)
        MFCC_A = librosa.feature.delta(MFCC, width=w, order=2)
    
    return MFCC, MFCC_D, MFCC_A


