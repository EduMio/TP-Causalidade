import os
for i in range(18):
	os.system('py train-preprocess.py ' + str(i))
os.system('py test-preprocess.py')