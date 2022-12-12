from common import*

BEG = '../data/TestData/'
dfLabels = pd.read_csv(BEG + 'annotations/gold_standard.csv')
dfAgeSex = pd.read_csv(BEG + 'attributes.csv')

with h5py.File(BEG + 'ecg_tracings.hdf5') as f:
    M = np.array(f['tracings'])

eletrodoIdx = 0
nPessoas = len(M)
c = []
for i in range(nPessoas):
    if i % 50 == 0:
        print(i, '/', nPessoas)
    x = M[i,:,eletrodoIdx]
    l = dfLabels.values[i]
    l[-2], l[-1] = l[-1], l[-2] # deixar as labels das doenças na mesma ordem de train
    a = dfAgeSex.values[i][0]
    try:
        for seg in SegmentPQRST(x):
            if np.any(np.isnan(seg)): # neurokit returned nan
                continue
            coeff = GetEachWaveCoefficients(x, seg)
            row = np.zeros(len(coeff) + 7)
            row[0] = a
            row[1:7] = l
            row[7:] = coeff
            c.append(row)
    except Exception as e:
        print(e, i)
c = np.array(c)
np.save('test.npy', c)