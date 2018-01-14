from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()

# Show the data (the attributes of each instance)
print(iris.data)

# Show the target values (in numeric format) of each instance
print(iris.target)

# Show the actual target names that correspond to each number
print(iris.target_names)

# Randomize the data and split it 70/30 (training/test)
data_train, data_test, targets_train, targets_test = \
    train_test_split(iris.data, iris.target, train_size=0.7, test_size=0.3, random_state=42)

# Use Naive Bayes algorithm to train the model
classifier = GaussianNB()
model = classifier.fit(data_train, targets_train)

# Use the model to predict target values
targets_predicted = model.predict(data_test)


# Defines a function that checks the accuracy of predicted values
def check_accuracy(prediction, real):
    correct = len([i for i, j in zip(prediction, real) if i == j])
    accuracy = correct/len(real)
    print("{}/{} correct - {:.1f}% accuracy".format(correct, len(real), accuracy*100))


# Check accuracy of the Naive Bayes model
check_accuracy(targets_predicted, iris.target)


# Define new classes for new algorithm
class HardCodedModel:
    def __init__(self, data=None, targets=None):
        self.data = data
        self.targets = targets

    def predict(self, data):
        prediction = []
        for i in self.data:
            prediction.append(0)
        return prediction


class HardCodedClassifier:
    def __init__(self):
        pass

    def fit(self, data=None, targets=None):
        return HardCodedModel(data, targets)


# Use new algorithm to create a model and make predictions
new_classifier = HardCodedClassifier()
new_model = new_classifier.fit(data_train, targets_train)
new_targets_predicted = new_model.predict(data_test)

# Check accuracy of the new model (it should be 33%)
check_accuracy(new_targets_predicted, iris.target)
