from numpy import count_nonzero, logical_and, logical_or, concatenate, mean, array_split, poly1d, polyfit, array
from numpy.random import permutation
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt


SVM_DEFAULT_DEGREE = 3
SVM_DEFAULT_GAMMA = 'auto'
SVM_DEFAULT_C = 1.0
ALPHA = 1.5


def prepare_data(data, labels, max_count=None, train_ratio=0.8):
    """
    :param data: a numpy array with the features dataset
    :param labels:  a numpy array with the labels
    :param max_count: max amout of samples to work on. can be used for testing
    :param train_ratio: ratio of samples used for train
    :return: train_data: a numpy array with the features dataset - for train
             train_labels: a numpy array with the labels - for train
             test_data: a numpy array with the features dataset - for test
             test_labels: a numpy array with the features dataset - for test
    """
    if max_count:
        data = data[:max_count]
        labels = labels[:max_count]

    # Combine data with labels in order to shuffle it.
    full_data = concatenate((data, array([labels]).T), axis=1)
    full_data = permutation(full_data)

    # Split the data into train and test set by the given ratio.
    limit = int(train_ratio * len(full_data))
    train, test = full_data[:limit], full_data[limit:]

    train_data = train[:,:-1]
    train_labels = train[:,-1]
    test_data = test[:,:-1]
    test_labels = test[:,-1]

    return train_data, train_labels, test_data, test_labels


def get_stats(prediction, labels):
    """
    :param prediction: a numpy array with the prediction of the model
    :param labels: a numpy array with the target values (labels)
    :return: tpr: true positive rate
             fpr: false positive rate
             accuracy: accuracy of the model given the predictions
    """

    length = len(labels)
    positives = labels.sum()
    negativies = length - positives

    # True classified positives / total true positives
    tp = logical_and(prediction, labels).sum()
    tpr = tp / positives

    # Falsy true classification/ total true negativies.
    fpr = array([prediction[i] and not labels[i] for i in range(length)]).sum()
    fpr /= negativies

    # True negative count
    tn = array([not prediction[i] and not labels[i] for i in range(length)]).sum()
    accuracy = (tp + tn) / length

    return tpr, fpr, accuracy


def get_k_fold_stats(folds_array, labels_array, clf):
    """
    :param folds_array: a k-folds arrays based on a dataset with M features and N samples
    :param labels_array: a k-folds labels array based on the same dataset
    :param clf: the configured SVC learner
    :return: mean(tpr), mean(fpr), mean(accuracy) - means across all folds
    """
    tpr = []
    fpr = []
    accuracy = []

    # Iterate each element in the k-folds array. Each iteration
    # the current elemnt is the testing element and the rest are
    # the training instances.
    for idx, test_el in enumerate(folds_array):
        # Cut the testing element out of the k-folds array.
        train = concatenate((folds_array[:idx] + folds_array[idx+1:]))
        labels = concatenate((labels_array[:idx] + labels_array[idx+1:]))

        clf.fit(train, labels) # Fit the SVM model.
        prediction = clf.predict(test_el) # Predict.

        # Evaluate current classifier preformance.
        _tpr, _fpr, _ac = get_stats(prediction, labels_array[idx])

        tpr.append(_tpr)
        fpr.append(_fpr)
        accuracy.append(_ac)

    return mean(tpr), mean(fpr), mean(accuracy)


def compare_svms(data_array,
                 labels_array,
                 folds_count,
                 kernels_list=('poly', 'poly', 'poly', 'rbf', 'rbf', 'rbf',),
                 kernel_params=({'degree': 2}, {'degree': 3}, {'degree': 4}, {'gamma': 0.005}, {'gamma': 0.05}, {'gamma': 0.5},)):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :param kernels_list: a list of strings defining the SVM kernels
    :param kernel_params: a dictionary with kernel parameters - degree, gamma, c
    :return: svm_df: a dataframe containing the results as described below
    """
    svm_df = pd.DataFrame()
    svm_df['kernel'] = kernels_list
    svm_df['kernel_params'] = kernel_params
    svm_df['tpr'] = None
    svm_df['fpr'] = None
    svm_df['accuracy'] = None

    # Helper arrays to hold the different k-fold stats results.
    tpr = []
    fpr = []
    accuracy = []

    partition_data = array_split(data_array, folds_count)
    partition_labels = array_split(labels_array, folds_count)

    # Iterate each kernel and generate a learner.
    for idx, kernel in enumerate(kernels_list):
        clf = SVC(
            C=kernel_params[idx].get('C') or SVM_DEFAULT_C,
            kernel=kernel,
            degree=kernel_params[idx].get('degree') or SVM_DEFAULT_DEGREE,
            gamma=kernel_params[idx].get('gamma') or SVM_DEFAULT_GAMMA,
        )

        _tpr, _fpr, _acc = get_k_fold_stats(partition_data, partition_labels, clf)

        # Append the current k-fold stats.
        tpr.append(_tpr)
        fpr.append(_fpr)
        accuracy.append(_acc)

    # We're done.
    svm_df['tpr'] = tpr
    svm_df['fpr'] = fpr
    svm_df['accuracy'] = accuracy

    return svm_df


def get_most_accurate_kernel(res):
    """
    :return: integer representing the row number of the most accurate kernel
    """
    return get_argmax(res.get('accuracy'))

def get_kernel_with_highest_score(res):
    """
    :return: integer representing the row number of the kernel with the highest score
    """
    return get_argmax(res.get('score'))


def plot_roc_curve_with_score(df, alpha_slope=1.5):
    """
    :param df: a dataframe containing the results of compare_svms
    :param alpha_slope: alpha parameter for plotting the linear score line
    :return:
    """
    x = df.fpr.tolist()
    y = df.tpr.tolist()

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def evaluate_c_param(data_array, labels_array, folds_count):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :return: res: a dataframe containing the results for the different c values. columns similar to `compare_svms`
    """

    res = pd.DataFrame()
    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return res


def get_test_set_performance(train_data, train_labels, test_data, test_labels):
    """
    :param train_data: a numpy array with the features dataset - train
    :param train_labels: a numpy array with the labels - train

    :param test_data: a numpy array with the features dataset - test
    :param test_labels: a numpy array with the labels - test
    :return: kernel_type: the chosen kernel type (either 'poly' or 'rbf')
             kernel_params: a dictionary with the chosen kernel's parameters - c value, gamma or degree
             clf: the SVM leaner that was built from the parameters
             tpr: tpr on the test dataset
             fpr: fpr on the test dataset
             accuracy: accuracy of the model on the test dataset
    """

    kernel_type = ''
    kernel_params = None
    clf = SVC(class_weight='balanced')  # TODO: set the right kernel
    tpr = 0.0
    fpr = 0.0
    accuracy = 0.0

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return kernel_type, kernel_params, clf, tpr, fpr, accuracy

def get_argmax(items):
    """
    Helper method to return the index of the highet value with a given ndarray.
    """
    maximum = 0.0
    max_idx = 0

    for idx, val in enumerate(items):
        if val > maximum:
            max_idx = idx
            maximum = val

    return max_idx
