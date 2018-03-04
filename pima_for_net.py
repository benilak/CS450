"""
Pima-indians-diabetes data set preprocessed for the NetClassifier.
"""

pima = pd.read_csv('pima-indians-diabetes.txt', header=None)
pima_data, pima_targets = np.hsplit(pima, [8])
pima_data = pd.DataFrame(normalize(pima_data))
pima_targets = pima_targets.iloc[:, 0]
data_train, data_test, targets_train, targets_test = train_test_split(pima_data, pima_targets, train_size=0.8)

classifier = NetClassifier(data_train, targets_train, [6])
neuralnet = classifier.build_net(cycles=800)
cycle_correct = pd.Series(neuralnet[1])
plot = cycle_correct.plot.line()
plt.show()
model = NetModel(neuralnet[0])
predictions = model.predict(data_test, targets_test)
print(predictions[1])