from rebuild import*

db = np.load('preprocessed0.npy')
print(db.shape)
idx = 453
x = db[idx]
print('age', x[0])
print('labels', x[1:7])
for w, t in zip(RebuildWaves(x[7:]), ['P', 'QRS', 'T', 'ALL']):
	plt.plot(w)
	plt.title(t)
	plt.show()