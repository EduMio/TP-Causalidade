import os

TRAIN_DIR = '..' + os.sep + 'data' + os.sep + 'TrainData'
PREPROCESSED_DATA = 'PreprocessedTrain'
ELETRODO_IDX = 0
SAMPLE_RATE = 400
N_COEFF_CMPLX = 64
MIN_AGE, MAX_AGE = 17, 91

TOT_BINS = MAX_AGE - MIN_AGE + 1
N_COEFF = N_COEFF_CMPLX * 2
INTERP_LEN = 127

import numpy as np, h5py, neurokit2 as nk, warnings, time, pandas as pd, math
from scipy.fft import rfft

def TryMkDirs(d):
	try:
		os.makedirs(d)
	except:
		pass

def MapAgeToBin(age):
	return min(max(age, MIN_AGE), MAX_AGE) - MIN_AGE
	
def GetInterpolatedWave(w, L):
	'''
	retorna o sinal w interpolado linearmente para L observações
	'''
	ret = np.zeros(L)
	FRAC = 1 / L
	frac = 0
	s = len(w) - 1
	for i in range(L):
		off = frac * s
		ioff = int(off)
		fract = off - ioff
		l = w[ioff]
		r = w[ioff + 1]
		ret[i] = (1-fract)*l+fract*r
		frac += FRAC
	return ret

def GetCoeff(w):
	ret = np.zeros(1 + N_COEFF)
	ret[0] = len(w)
	interpW = GetInterpolatedWave(w, INTERP_LEN)
	c = rfft(interpW)
	ret[1::2] = c.real
	ret[2::2] = c.imag
	return ret

def FilterNeurokitWarnings():
	# filter some (bugs?) from neurokit2
	warnings.filterwarnings(action='ignore', message='Mean of empty slice')
	warnings.filterwarnings(action='ignore', message='invalid value encountered in double_scalars')
	warnings.filterwarnings(action='ignore', message='Too few peaks detected to compute the rate. Returning empty vector.')