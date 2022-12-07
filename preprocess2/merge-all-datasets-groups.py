from common import*

IN_DIR = 'AgeGroups'
OUT_DIR = 'AllAgeGroups'

AGE_OFF = 5
AGE_LIM_EXCLUSIVE = MAX_AGE + 1

for ageStart in range(MIN_AGE, AGE_LIM_EXCLUSIVE, AGE_OFF):
	allEcgs = []
	allLabels = []
	allMappings = []
	allOndas = []
	accSum = 0

	ageEnd = ageStart + AGE_OFF
	if ageEnd > AGE_LIM_EXCLUSIVE:
		ageEnd = AGE_LIM_EXCLUSIVE
	ageGroup = os.sep + str(ageStart) + '-' + str(ageEnd - 1) + os.sep

	for testDir in range(18):

		print(testDir, ageGroup)

		startIn = IN_DIR + os.sep + str(testDir) + ageGroup
		ecgs = np.load(startIn + 'ecgs.npy')
		mappings = np.load(startIn + 'mappings.npy')
		labels = np.load(startIn + 'labels.npy')
		ondas = np.load(startIn + 'ondas.npy')

		mappings += accSum
		accSum += len(mappings)
	
		for ecg in ecgs:
			allEcgs.append(ecg)
		for label in labels:
			allLabels.append(label)
		for onda in ondas:
			allOndas.append(onda)
		for mapping in mappings:
			allMappings.append(mapping)
	
	allEcgs = np.array(allEcgs)
	allLabels = np.array(allLabels)
	allOndas = np.array(allOndas)
	allMappings = np.array(allMappings)

	startOut = OUT_DIR + ageGroup
	TryMkDirs(startOut)

	np.save(startOut + 'ecgs', allEcgs)
	np.save(startOut + 'labels', allLabels)
	np.save(startOut + 'ondas', allOndas)
	np.save(startOut + 'mappings', allMappings)