import pandas as pd
import numpy as np
import KNearestNeighbors.knn as knn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier


# cars = pd.read_csv("cardata.txt", header=None)
# pima = pd.read_csv("..\\NeuralNet\\pima-indians-diabetes.txt", header=None)
# auto = pd.read_csv("auto-mpg.txt", header=None)

def get_cars_data():
    cars = pd.read_csv("cardata.txt", header=None)
    for i in range(len(cars.T) - 1):
        cars[i] = pd.Categorical(cars[i])
        cars[i] = cars[i].cat.codes
    cars = np.asarray(cars)
    data, targets = np.hsplit(cars, [6])
    data = normalize(data, axis=0)
    return data, targets

def get_pima_data():
    pima = pd.read_csv("..\\NeuralNet\\pima-indians-diabetes.txt", header=None)
    median = pima.median(axis=0)
    for i in range(len(pima.T) - 1):
        pima[i] = pima[i].replace(0, median[i])
    pima = np.asarray(pima)
    data, targets = np.hsplit(pima, [8])
    data = normalize(data, axis=0)
    return data, targets


def get_auto_data():
    auto = pd.read_csv("auto-mpg.txt", header=None, delim_whitespace=True)
    targets = np.asarray(auto[0])
    col = [0,8]
    auto = auto.drop(auto.columns[col], axis=1)
    mean = round(auto.mean(axis=0)[2])
    auto[3] = auto[3].replace('?', mean)
    data = normalize(auto, axis=0)
    # data = auto
    return data, targets

# get processed data and targets for each data set
cars_data, cars_targets = get_cars_data()
pima_data, pima_targets = get_pima_data()
auto_data, auto_targets = get_auto_data()

# split into training and testing sets
cars_data_train, cars_data_test, cars_targets_train, cars_targets_test = \
    train_test_split(cars_data, cars_targets, train_size=2/3)
pima_data_train, pima_data_test, pima_targets_train, pima_targets_test = \
    train_test_split(pima_data, pima_targets, train_size=2/3)
auto_data_train, auto_data_test, auto_targets_train, auto_targets_test = \
    train_test_split(auto_data, auto_targets, train_size=2/3)

# test our knn algorithm
print('cars_classifier')
cars_classifier = knn.KNNClassifier()
cars_model = cars_classifier.fit(cars_data_train, cars_targets_train)
cars_predictions = cars_model.predict(cars_data_test, k=3, d=2)
knn.check_accuracy(cars_predictions, cars_targets_test)

print('pima_classifier')
pima_classifier = knn.KNNClassifier()
pima_model = pima_classifier.fit(pima_data_train, pima_targets_train)
pima_predictions = pima_model.predict(pima_data_test, k=3, d=2)
knn.check_accuracy(pima_predictions, pima_targets_test)

print('auto_classifier')
auto_classifier = knn.KNNClassifier()
auto_model = auto_classifier.fit(auto_data_train, auto_targets_train)
auto_predictions = auto_model.predict(auto_data_test, k=3, d=2)
knn.check_accuracy(auto_predictions, auto_targets_test)

# compare to existing knn implementation
print('xcars_classifier')
xcars_classifier = KNeighborsClassifier(n_neighbors=2)
xcars_model = xcars_classifier.fit(cars_data_train, cars_targets_train)
xcars_predictions = xcars_model.predict(cars_data_test)
knn.check_accuracy(xcars_predictions, cars_targets_test)

print('xpima_classifier')
xpima_classifier = KNeighborsClassifier(n_neighbors=2)
xpima_model = xpima_classifier.fit(pima_data_train, pima_targets_train)
xpima_predictions = xpima_model.predict(pima_data_test)
knn.check_accuracy(xpima_predictions, pima_targets_test)

print('xauto_classifier')
xauto_classifier = KNeighborsClassifier(n_neighbors=2)
xauto_model = xauto_classifier.fit(auto_data_train, auto_targets_train)
xauto_predictions = xauto_model.predict(auto_data_test)
knn.check_accuracy(xauto_predictions, auto_targets_test)

