import os, time

t0 = time.time()

for i in range(18):
	os.system('py train-preprocess.py ' + str(i))
os.system('py test-preprocess.py')

print('Took', time.time() - t0, 'seconds')