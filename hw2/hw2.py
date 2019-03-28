import numpy as np
np.random.seed(42)

chi_table = {0.01  : 6.635,
             0.005 : 7.879,
             0.001 : 10.828,
             0.0005 : 12.116,
             0.0001 : 15.140,
             0.00001: 19.511}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.
    """
    if len(data) < 2:
        return 0

    dict, total = count_labels(data)

    gini = 0.0
    for key in dict:
        gini += np.square(dict[key] / total)

    return 1 - gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.
    """
    if len(data) < 2:
        return 0

    dict, total = count_labels(data)

    entropy = 0.0
    for key in dict:
        p = dict[key] / total

        entropy -= (p * np.log2(p))

    # print('final entropy: ', entropy)
    return entropy

class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic
    # functionality as described in the notebook. It is highly recommended that you
    # first read and understand the entire exercise before diving into this class.

    def __init__(self, feature, value):
        self.feature = feature # column index of criteria being tested
        self.value = value # value necessary to get a true result
        self.children = []
        self.summary = None

    def add_child(self, node):
        self.children.append(node)

    def learn(self, data, impurity, chi_value=1):
        """
        Given an impurity measure and a data set, choose the best attribute
        that will bring us closer to perfect classification by creating child
        nodes based on impurity decisions.
        """
        # Save on the node the number of appearences for each class
        self.summary = count_labels(data)

        if (impurity(data) == 0):
            # We're done with current node.
            return

        max_gain = 0
        for idx in range(data.shape[1] - 1):
            thresholds = get_thresholds(data, idx)
            thresh, gain = find_best_val(data, idx, thresholds, impurity)

            if (gain > max_gain):
                max_gain = gain
                self.value = thresh
                self.feature = idx

        # Split data by resulted attribute threshold and add
        # the new children to the current node.
        d1, d2 = split_by_threshold(data, self.feature, self.value)

        prune = False
        if chi_value in chi_table:
            print("chi_value in chi_table.")
            res = self.calc_chi(d1, d2)
            com = chi_table[chi_value]
            print("calculated chi: ", res)
            prune = res < com

        if prune:
            print('Should prune.')
            # Prediction power is not strong enough for splitting
            # according to the attribute and is more towards
            # random distribution.
            return

        left_child = DecisionNode(-1, -1)
        right_child = DecisionNode(-1, -1)
        self.add_child(left_child)
        self.add_child(right_child)

        # Learn upon new dispresion
        left_child.learn(d1, impurity)
        right_child.learn(d2, impurity)

    def predict(self, instance):
        if len(self.children) < 1:
            #this is a leaf node, we need to decide a prediction
            labels = self.summary[0]
            return max(labels.keys(), key=lambda k: labels[k])

        child = None
        if instance[self.feature] > self.value:
            child = self.children[0]
        else:
            child = self.children[1]

        return child.predict(instance)

    def calc_chi(self, d0, d1):
        labels = self.summary[0]
        total = self.summary[1]

        # Defensive counting
        if 0 not in labels:
            labels[0] = 0
        if 1 not in labels:
            labels[1] = 0

        py0 = labels[0] / total
        py1 = labels[1] / total

        pf0 = (d0[:, -1] == 0).sum()
        nf0 = (d0[:, -1] == 1).sum()
        e00 = len(d0) * py0
        e01 = len(d0) * py1

        pf1 = (d1[:, -1] == 0).sum()
        nf1 = (d1[:, -1] == 1).sum()
        e10 = len(d1) * py0
        e11 = len(d1) * py1

        result = (np.square(pf0 - e00) / e00) + (np.square(nf0 - e01) / e01)
        result +=(np.square(pf1 - e10) / e10) + (np.square(nf1 - e11) / e11)
        return result

    def __str__(self):
        return "Feature: {0}, value: {1}, # of children: {2}".format(
            self.feature, self.value, len(self.children)
        )


def build_tree(data, impurity, chi_value):
    """
    Build a tree using the given impurity measure and training dataset.
    You are required to fully grow the tree until all leaves are pure.

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    root = DecisionNode(-1, -1)
    root.learn(data, impurity, chi_value=chi_value)

    return root


def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    return node.predict(instance)

def calc_accuracy(node, dataset):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0.0
    good_predictions = 0

    for ins in dataset:
        prediction = predict(node, ins)

        if prediction == ins[-1]:
            good_predictions += 1

    return (good_predictions / dataset.shape[0]) * 100


def print_tree(node):
    '''
    prints the tree according to the example in the notebook

	Input:
	- node: a node in the decision tree

	This function has no return value
	'''

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

def count_labels(data):
    """
    Input: a data set whose last column is target labels.
    Returns a dictionary representing number of appearences for each class
    and the total number of instances.
    """
    dict = {} # Counts appearences of labels/classes.
    labels = data[:,-1]

    for label in labels:
        if label not in dict:
            dict[label] = 1
        else:
            dict[label] += 1

    return dict, len(labels)

def get_thresholds(data, attr_idx):
    """
    Given a data set and an attribute (column index), return list of thresholds.
    """
    thresholds = []
    attribute = np.sort(data[:,attr_idx], kind='mergesort')

    for idx, val in enumerate(attribute):
        thresholds.append((val + attribute[idx + 1]) / 2)

        if idx + 2 == len(attribute):
            break;

    return thresholds

def find_best_val(data, attr_idx, thresholds, impurity):
    """
    Given a data set, attribute (column index) and a list of thresholds,
    find the best value that brings us closer to perfect classification.
    """
    max_impuriy_reduce = 0
    best_threshold = None
    total = len(data)
    current_impurity = impurity(data)

    for threshold in thresholds:
        # Create at most a single split for each node of the tree
        d1, d2 = split_by_threshold(data, attr_idx, threshold)

        sum = (len(d1) / total) * impurity(d1)
        sum += (len(d2) / total) * impurity(d2)

        # Choose the attribute that cause the largest impurity reduce
        if (current_impurity - sum > max_impuriy_reduce):
            max_impuriy_reduce = current_impurity - sum
            best_threshold = threshold

    return best_threshold, max_impuriy_reduce

def split_by_threshold(data, attr_idx, threshold):
    """
    Given a data set, an attribute (column index) and a threshold value,
    split the data by that value and return it as two sets.
    """
    d1 = []
    d2 = []

    for d in data:
        if (d[attr_idx] > threshold):
            d1.append(d)
        else:
            d2.append(d)

    return np.array(d1), np.array(d2)
