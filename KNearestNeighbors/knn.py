import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier


class KNNModel:
    def __init__(self, training_data, training_targets):
        self.training_data = training_data
        self.training_targets = training_targets
        # self.nfeatures = len(self.training_data.columns)  # used for calc_distance

    # this isn't used (it's slower, but I'm keeping it here just in case)
    # def calc_distance(self, test_point, train_point, dim=1):
    #     sum = 0
    #     for i in range(self.nfeatures):
    #         sum += (abs(test_point[i] - train_point[i])) ** dim
    #     distance = sum ** (1 / dim)
    #     return distance


    def predict(self, test_data, k, d=1):

        ntest_points = np.shape(test_data)[0]

        # create empty list to fill with the class prediction for each test_point
        nearest = np.zeros(ntest_points)

        # predict class for each test_point in the test_data
        for n in range(ntest_points):
            # sums the distance between features for each train_point-test_point pair - this is total "distance"
            distances = (np.sum((abs(self.training_data - test_data[n, :])**d), axis=1))**(1/d)

            # create a list 'classes' of the classes of the nearest neighbors
            indices = np.argsort(distances, axis=0)
            classes = np.unique(self.training_targets[indices[:k]])

            # if there is only one nearest neighbor, its class is the prediction for the nth test_point
            if len(classes) == 1:
                nearest[n] = np.unique(classes)

            # count the number of instances for each class of the nearest neighbors
            else:
                counts = np.zeros(max(classes) + 1)
                for i in range(k):
                    counts[self.training_targets[indices[i]]] += 1

                # choose the most frequent class as the prediction for the nth test_point
                nearest[n] = np.max(counts)
        return nearest


class KNNClassifier:
    def __init__(self):
        pass

    def fit(self, training_data, training_targets):
        return KNNModel(training_data, training_targets)


# Defines a function that checks the accuracy of predicted values
def check_accuracy(prediction, real):
    correct = len([i for i, j in zip(prediction, real) if i == j])
    accuracy = correct/len(real)
    print("{}/{} correct - {:.1f}% accuracy".format(correct, len(real), accuracy*100))


# load iris data set and normalize the numeric values
iris = datasets.load_iris()
iris_data = normalize(iris.data, axis=1)

# partition the data into training and testing subsets
data_train, data_test, targets_train, targets_test = \
    train_test_split(iris_data, iris.target, train_size=2/3)

# test our knn algorithm
knn_classifier = KNNClassifier()
knn_model = knn_classifier.fit(data_train, targets_train)
knn_predictions = knn_model.predict(data_test, k=3, d=2)
check_accuracy(knn_predictions, targets_test)

# compare to existing knn implementation
ex_classifier = KNeighborsClassifier(n_neighbors=2)
ex_model = ex_classifier.fit(data_train, targets_train)
ex_predictions = ex_model.predict(data_test)
check_accuracy(ex_predictions, targets_test)
