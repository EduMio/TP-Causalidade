from common import*

def Preprocess(fileInt, eletrodoIdx):
	'''
	dado o caminho do arquivo exams.csv, do .hdf5, o eletrodoIdx, computa o vetor
	[age, 6 labels, coeficientes P, QRS, T, Todos]
	'''
	
	p = 'exams_part' + str(fileInt) + '.hdf5'
	with h5py.File(p) as f:
		examsIds = np.array(f['exam_id'])
		M = np.array(f['tracings'])
	# descartar erro no dataset
	M = M[:-1]
	examsIds = examsIds[:-1]
	
	df = pd.read_csv('exams.csv')
	df = df[df['trace_file'] == p]
	ill = df[(df['1dAVb']|df['RBBB']|df['LBBB']|df['SB']|df['ST']|df['AF'])]
	healthy = df[~(df['1dAVb']|df['RBBB']|df['LBBB']|df['SB']|df['ST']|df['AF'])]
	healthy = healthy.sample(min(len(healthy), len(ill)), random_state=SEED)
	
	print('going to process', len(ill), 'ill people and', len(healthy), 'healthy')
	
	s = set(ill.exam_id.values)
	s.update(healthy.exam_id.values)
	
	ret = []
	nPessoas = len(M)
	for i in range(nPessoas):
		examId = examsIds[i]
		if examId in s:
			x = M[i,:,eletrodoIdx]
			try:
				for seg in SegmentPQRST(x):
					if np.any(np.isnan(seg)): # neurokit returned nan
						continue
					coeff = GetEachWaveCoefficients(x, seg)
					row = df[df['exam_id'] == examId]
					def getE(s):
						return row[s].values[0]
					tot = np.zeros(len(coeff) + 8)
					tot[:8] = [examId,
						getE('age'),
						getE('1dAVb'),
						getE('RBBB'),
						getE('LBBB'),
						getE('SB'),
						getE('ST'),
						getE('AF')
					]
					tot[8:] = coeff
					ret.append(tot)
			except Exception as e:
				print(e, i)
		if i % 100 == 0:
			print(i, '/', nPessoas)
	ret = np.array(ret)
	return ret

l = sys.argv
if len(l) < 2:
	print('Passe o número do arquivo teste a ser preprocessado [0, 17]')
else:
	t0 = time.time()
	fileInt = int(l[1])
	c = Preprocess(fileInt, 0)
	now = time.time()
	np.save('preprocessed' + str(fileInt), c)
	print('Took', now-t0)