import os, threading, time

t0 = time.time()

def f(cmd):
	os.system(cmd)

threads = []
for i in range(18):
	threads.append(threading.Thread(target=f, args=('py train-preprocess.py ' + str(i), )))
threads.append(threading.Thread(target=f, args=('py test-preprocess.py', )))
for i in threads:
	i.start()
for i in threads:
	i.join()

print('Took', time.time() - t0, 'seconds')