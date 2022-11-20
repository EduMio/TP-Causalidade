import numpy as np, matplotlib.pyplot as plt, h5py, pandas as pd, os, neurokit2 as nk, warnings, time, sys
from scipy.fft import rfft, irfft

# filter some (bugs?) from neurokit2
warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore', message='invalid value encountered in double_scalars')
warnings.filterwarnings(action='ignore', message='Too few peaks detected to compute the rate. Returning empty vector.')

SEED=0
SAMPLE_RATE = 400

# https://en.wikipedia.org/wiki/Electrocardiography#:~:text=on%20the%20source.-,Amplitudes%20and%20intervals,-%5Bedit%5D
# A lógica usada para decidir o número de coeficientes de cada onda é: proporcional à sua duração máxima em geral
# P : 80ms
# QRS: 100ms
# T: 160ms

nP = 16
nQRS = nP * 100 // 80
nT = nP * 160 // 80
nAll = 64

def GetWaveLen(x):
	'''
	retorna o número de observações em sinal para que seu rfft tenha x coeficientes
	'''
	return x*2-1

L_P = GetWaveLen(nP)
L_QRS = GetWaveLen(nQRS)
L_T = GetWaveLen(nT)
L_All = GetWaveLen(nAll)

tP = nP*2
tQRS = nQRS*2
tT = nT*2
tAll = nAll*2
TOT = 4+tP+tQRS+tT+tAll

def SegmentPQRST(x):
	'''
	x : onda completa (com vários períodos)
	retorna lista, onde cada elemento é:
	(início P, fim P, início QRS, fim QRS, início T, fim T, fim do período(=início da próxima P))
	função pode lançar exceção
	'''
	d = nk.ecg_delineate(x, sampling_rate=SAMPLE_RATE)[1]
	LABELS = ['ECG_P_Onsets', 'ECG_P_Offsets', 'ECG_R_Onsets', 'ECG_R_Offsets', 'ECG_T_Onsets', 'ECG_T_Offsets']
	fLabel = LABELS[0]
	n = len(d[fLabel])
	ret = []
	for i in range(n-1):
		cur = []
		for l in LABELS:
			cur.append(d[l][i])
		cur.append(d[fLabel][i+1])
		ret.append(cur)
	return ret

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

def GetEachWaveCoefficients(x, seg):
	'''
	x : ecg inteiro
	seg : lista de 7 elementos, contendo a segmentação de uma onda no ecg
	retorna um vetor de TOT elementos na ordem (P, QRS, T, completa)
	[len(P), coeficientes de P, len(QRS), coeficientes de QRS, len(T), coef T, len(ALL), coeff ALL]
	'''
	ret = np.zeros(TOT+2)
	s = 0
	mean = np.mean(x[seg[0]:seg[-1]])
	def saveWaveAndLen(a, b, newLen):
		nonlocal ret, s
		a, b = seg[a], seg[b]
		ret[s] = b-a
		w = x[a:b] - mean
		interpW = GetInterpolatedWave(w, newLen)
		c = rfft(interpW)
		l = len(c)
		ret[s+1:s+1+l*2:2] = c.real
		ret[s+2:s+2+l*2:2] = c.imag
		s += 1 + l * 2
	for i, l in zip(range(3), [L_P, L_QRS, L_T]):
		saveWaveAndLen(i*2, i*2+1, l)
	saveWaveAndLen(0, 5, L_All)
	ret[s] = seg[2]-seg[1]
	ret[s+1] = seg[4]-seg[3]
	return ret