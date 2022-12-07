from common import*

JOINED_DIR = 'AgeGroups'

AGE_OFF = 5
AGE_LIM_EXCLUSIVE = MAX_AGE + 1

for testDir in range(18):
	for ageStart in range(MIN_AGE, AGE_LIM_EXCLUSIVE, AGE_OFF):
		ageEnd = ageStart + AGE_OFF
		if ageEnd > AGE_LIM_EXCLUSIVE:
			ageEnd = AGE_LIM_EXCLUSIVE
		allEcgs = []
		allLabels = []
		allMappings = []
		allOndas = []
		accSum = 0
		DIR_OFF = os.sep + str(testDir) + os.sep
		for age in range(ageStart, ageEnd):
			print(testDir, age)
			start = PREPROCESSED_DATA + DIR_OFF + str(age) + os.sep
			ecgs = np.load(start + 'ecgs.npy')
			mappings = np.load(start + 'mappings.npy')
			labels = np.load(start + 'labels.npy')
			ondas = np.load(start + 'ondas.npy')

			mappings += accSum
			accSum += len(ecgs)
	
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
	
		start = JOINED_DIR + DIR_OFF + str(ageStart) + '-' + str(ageEnd - 1) + os.sep

		TryMkDirs(start)

		np.save(start + 'ecgs', allEcgs)
		np.save(start + 'labels', allLabels)
		np.save(start + 'ondas', allOndas)
		np.save(start + 'mappings', allMappings)