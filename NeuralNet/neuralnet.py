import pandas as pd
import numpy as np
from random import uniform
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import math


def check_accuracy(predictions, targets):
    """
    Checks the accuracy of predicted values.

    :param list predictions: a list of the predicted classes for the data
    :param pd.Series targets: the testing targets used for determining the model's accuracy
    :return string: a statement describing the accuracy of the model
    """

    correct = len([i for i, j in zip(predictions, targets) if i == j])
    accuracy = correct / len(targets)
    return "{}/{} correct - {:.1f}% accuracy".format(correct, len(targets), accuracy * 100)


class Node:
    def __init__(self, value=None, error=None, weights=None):
        """
        A class for a single node (within a particular layer) of a neural net.

        :param float value: the input value or activation value of the node, depending on the node type
        :param float error: the error value of the node
        :param list weights: a list of weights that are mapped to the node from the previous layer
        """

        self.value = value
        self.error = error
        self.weights = weights


class Layer:
    def __init__(self, nodes=None):
        """
        A class for a single layer of a neural net.

        :param list nodes: a list of the nodes within the layer
        """
        self.nodes = nodes


class NetModel:
    def __init__(self, net):
        """
        A neural net, which can make predictions for the classification of given data.

        :param list net: a list of layers from the NetClassifier, which is a description of the neural net
        """

        self.net = net

    def predict(self, data, targets):
        """
        Makes predictions for the testing data.
        The code itself is essentially just a subset of the training algorithm in the NetClassifier

        :param pd.DataFrame data: the testing data used to make predictions
        :param pd.Series targets: the testing targets used for determining the model's accuracy
        :return pd.Series: a series (to be consistent with the training targets) containing the predictions
        """

        data.insert(0, 'Bias', [-1 for i in range(data.shape[0])])
        classes = targets.unique()
        classes.sort()
        predictions = []

        # for each row in the data, iterate over each layer
        for r in range(data.shape[0]):

            # FEED FORWARD through each layer
            for i, layer in enumerate(self.net):
                # set values for the input nodes
                if i == 0:
                    for j, node in enumerate(layer.nodes):
                        node.value = data.iloc[r, j]
                # update values of the remaining nodes
                else:
                    for node in layer.nodes:
                        zipp = zip([node.value for node in self.net[i - 1].nodes], node.weights)
                        product = [j * k for j, k in zipp]
                        node.value = 1 / (1 + math.exp(-math.fsum(product)))

            prediction = classes[np.argmax([node.value for node in self.net[-1].nodes])]
            predictions.append(prediction)

        accuracy = check_accuracy(predictions, targets)

        return predictions, accuracy


class NetClassifier:
    def __init__(self, data, targets, hidden_shape=None):
        """
        Establishes the base structure of the neural net that can be built using 'build_net()'.

        :param pd.DataFrame data: the training data used to build the net
        :param pd.Series targets: the training targets used to build the net
        :param list hidden_shape: a list containing the desired number of nodes for each hidden layer
        """

        # initialize the data
        self.data = data
        self.data.insert(0, 'Bias', [-1 for i in range(data.shape[0])])
        self.targets = targets
        self.classes = self.targets.unique()
        self.classes.sort()

        # build layers and nodes
        self.layers = [Layer() for i in (range(len(hidden_shape) + 2))]
        self.layers[0].nodes = [Node() for i in range(self.data.shape[1])]
        for i, number in enumerate(hidden_shape):
            self.layers[1 + i].nodes = [Node() for j in range(number)]
        self.layers[-1].nodes = [Node() for i in range(len(self.classes))]

        # initialize weights between the nodes
        for i, layer in enumerate(self.layers):
            # the input layer's nodes have no incoming weights
            if i == 0:
                pass
            else:
                for node in layer.nodes:
                    node.weights = [uniform(-1/math.sqrt(len(self.layers[i-1].nodes)),
                                            1/math.sqrt(len(self.layers[i-1].nodes)))
                                    for n in range(len(self.layers[i - 1].nodes))]

    def build_net(self, rate=0.2, cycles=1000):
        """
        Builds the neural net.

        :param float rate: the learning rate used in this algorithm
        :param int cycles: the number of times to iterate over the entire data set
        :return:
        """

        # run the algorithm over the entire data set many times
        for cycle in range(cycles):
            # for each row in the data, iterate over each layer
            for r in range(self.data.shape[0]):

                # FEED FORWARD through each layer
                for i, layer in enumerate(self.layers):
                    # set values for the input nodes
                    if i == 0:
                        for j, node in enumerate(layer.nodes):
                            node.value = self.data.iloc[r, j]
                    # update values for the hidden nodes
                    elif i != (len(self.layers) - 1):
                        for n, node in enumerate(layer.nodes):
                            if n == 0:
                                node.value = -1
                            else:
                                values = [node.value for node in self.layers[i - 1].nodes]
                                v_zip = zip(values, node.weights)
                                v_product = [j * k for j, k in v_zip]
                                node.value = 1 / (1 + math.exp(-math.fsum(v_product)))
                    else:
                        for n, node in enumerate(layer.nodes):
                            values = [node.value for node in self.layers[i - 1].nodes]
                            v_zip = zip(values, node.weights)
                            v_product = [j * k for j, k in v_zip]
                            node.value = 1 / (1 + math.exp(-math.fsum(v_product)))

                # PROPAGATE BACK through each layer
                for i, layer in reversed(list(enumerate(self.layers))):
                    # calculate output errors
                    if i == (len(self.layers) - 1):
                        for j, node in enumerate(layer.nodes):
                            target = int(self.classes[j] == self.targets.iloc[r])
                            node.error = node.value * (1 - node.value) * (node.value - target)
                            node.weights = [(weight - rate * node.error * self.layers[i-1].nodes[q].value)
                                            for q, weight in enumerate(node.weights)]
                    # calculate errors for the layer preceding the output layer
                    elif i == (len(self.layers) - 2):
                        for j, node in enumerate(layer.nodes):
                            if j == 0:
                                pass
                            else:
                                errors = [node.error for node in self.layers[i + 1].nodes]
                                weights = [node.weights[j] for node in self.layers[i + 1].nodes]
                                e_zip = zip(errors, weights)
                                e_product = [m * n for m, n in e_zip]
                                node.error = node.value * (1 - node.value) * math.fsum(e_product)
                                node.weights = [(weight - rate * node.error * self.layers[i - 1].nodes[q].value)
                                                for q, weight in enumerate(node.weights)]
                    # calculate errors for all other hidden layers
                    elif i != 0:
                        for j, node in enumerate(layer.nodes):
                            if j == 0:
                                pass
                            else:
                                errors = [node.error for node in self.layers[i + 1].nodes[1:]]
                                weights = [node.weights[j] for node in self.layers[i + 1].nodes[1:]]
                                e_zip = zip(errors, weights)
                                e_product = [m * n for m, n in e_zip]
                                node.error = node.value * (1 - node.value) * math.fsum(e_product)
                                node.weights = [(weight - rate * node.error * self.layers[i - 1].nodes[q].value)
                                                for q, weight in enumerate(node.weights)]
                    # the input nodes do not have errors or weights
                    else:
                        pass
            print(cycle)
        return self.layers
