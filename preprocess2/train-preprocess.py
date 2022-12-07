from common import*
import sys

l = sys.argv
if len(l) < 2:
	print('Passe o número do arquivo teste a ser preprocessado [0, 17]')
	quit()

t0 = time.time()

fileIdx = int(l[1])

BEG = PREPROCESSED_DATA + os.sep + str(fileIdx)

for i in range(17, 92):
	TryMkDirs(BEG + os.sep + str(i))

# filter some (bugs?) from neurokit2
warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore', message='invalid value encountered in double_scalars')
warnings.filterwarnings(action='ignore', message='Too few peaks detected to compute the rate. Returning empty vector.')

p = 'exams_part' + str(fileIdx) + '.hdf5'
with h5py.File(TRAIN_DIR + os.sep + p) as f:
	examsIds = np.array(f['exam_id'])
	M = np.array(f['tracings'])
# descartar erro no dataset
M = M[:-1]
examsIds = examsIds[:-1]

df = pd.read_csv(TRAIN_DIR + os.sep + 'exams.csv')
df = df[df['trace_file'] == p]

def GetDFRow(i):
	examId = examsIds[i]
	return df[df['exam_id'] == examId]

def GetRowValue(row, key):
	return row[key].values[0]

def GetLabelsFromRow(row):
	LABELS = ['1dAVb', 'RBBB', 'LBBB', 'SB' , 'ST', 'AF']
	return np.array([GetRowValue(row, label) for label in LABELS])

ecgs = [[] for _ in range(TOT_BINS)]
ondas = [[] for _ in range(TOT_BINS)]
mappings = [[] for _ in range(TOT_BINS)] # mapeia a onda para o ecg de que foi retirado
labels = [[] for _ in range(TOT_BINS)]

nPessoas = len(M)
for i in range(nPessoas):
	
	if i % 50 == 0:
		print(i, '/', nPessoas)	
		
	ecg = M[i, :, ELETRODO_IDX]
	
	def TryDelineate(w):
		try:
			d = nk.ecg_delineate(w, sampling_rate=SAMPLE_RATE)[1]
			return d
		except:
			print("Couldn't delineate", i)
			return None
		
	w = ecg - ecg.mean()
	d = TryDelineate(w)
	if d is not None:

		row = GetDFRow(i)
		age = GetRowValue(row, 'age')

		off = MapAgeToBin(age)
		starts = d['ECG_P_Onsets']
		found = False
		ecgsOff = ecgs[off]
		curRow = len(ecgsOff)
		for j in range(len(starts) - 1):
			cur = starts[j]
			nxt = starts[j + 1]
			if math.isnan(cur) or math.isnan(nxt):
				# print('Neurokit retornou nan', i)
				continue
				
			period = w[cur:nxt]
			if not len(period):
				print('Neurokit retornou período vazio', i)
				break

			c = GetCoeff(period)
			ondas[off].append(c)
			mappings[off].append(curRow)
			found = True
		if found:
			ecgs[off].append(ecg)
			labels[off].append(GetLabelsFromRow(row))

def ToNP(x):
	return [np.array(i) for i in x]
ecgs = ToNP(ecgs)
ondas = ToNP(ondas)

for i in range(TOT_BINS):
	start = BEG + os.sep + str(17 + i) + os.sep
	np.save(start + 'ecgs', ecgs[i])
	np.save(start + 'mappings', mappings[i])
	np.save(start + 'ondas', ondas[i])
	np.save(start + 'labels', labels[i])

print('Took', time.time() - t0, 'seconds')