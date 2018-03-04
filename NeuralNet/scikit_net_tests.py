"""
Using the MLPClassifier from scikit-learn on iris and pima to compare results.
The results were similar to the custom built NetClassifier.
"""

# iris = pd.read_csv('iris.txt')
# trash, iris_data, iris_targets = np.hsplit(iris, [1, 5])
# iris_data = pd.DataFrame(normalize(iris_data))
# iris_targets = iris_targets.iloc[:, 0]
# data_train, data_test, targets_train, targets_test = train_test_split(iris_data, iris_targets, train_size=2/3)
#
# pima = pd.read_csv('pima-indians-diabetes.txt', header=None)
# pima_data, pima_targets = np.hsplit(pima, [8])
# pima_data = pd.DataFrame(normalize(pima_data))
# pima_targets = pima_targets.iloc[:, 0]
# data_train, data_test, targets_train, targets_test = train_test_split(pima_data, pima_targets, train_size=0.8)

classifier = MLPClassifier((4))
classifier.fit(data_train, targets_train)
predictions = classifier.predict(data_test)
print(check_accuracy(predictions, targets_test))