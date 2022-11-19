from rebuild import*

db = np.load('test.npy')
print(db.shape)
idx = 0
x = db[idx]
print('age', x[0])
print('labels', x[1:7])
for w, t in zip(RebuildWaves(x[7:]), ['P', 'QRS', 'T', 'ALL']):
	plt.plot(w)
	plt.title(t)
	plt.show()