"""
Iris data set preprocessed for the NetClassifier.
"""

iris = pd.read_csv('iris.txt')
trash, iris_data, iris_targets = np.hsplit(iris, [1, 5])
iris_data = pd.DataFrame(normalize(iris_data))
iris_targets = iris_targets.iloc[:, 0]
data_train, data_test, targets_train, targets_test = train_test_split(iris_data, iris_targets, train_size=2/3)

classifier = NetClassifier(data_train, targets_train, [4])
neuralnet = classifier.build_net(cycles=100)
cycle_correct = pd.Series(neuralnet[1])
plot = cycle_correct.plot.line()
plt.show()
model = NetModel(neuralnet[0])
predictions = model.predict(data_test, targets_test)
print(predictions[1])
