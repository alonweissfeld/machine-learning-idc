import numpy as np
np.random.seed(42)

####################################################################################################
#                                            Part A
####################################################################################################

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.class_col = dataset[:, -1]
        self.class_value = class_value

        class_data = dataset[dataset[:, -1] == class_value][:, :-1]
        self.mean = np.mean(class_data, axis=0)
        self.std = np.std(class_data, axis=0)


    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        unique, counts = np.unique(self.class_col, return_counts=True)
        stats = dict(zip(unique, counts))

        return stats[self.class_value] / sum(stats.values())

    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """

        return np.prod(normal_pdf(x, self.mean, self.std))
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()
    
class MultiNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        class_data = dataset[dataset[:, -1] == class_value][:, :-1]
        self.mean = np.mean(class_data, axis=0)
        self.class_col = dataset[:, -1]
        self.class_value = class_value
        self.cov = np.cov(class_data.T)
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        unique, counts = np.unique(self.class_col, return_counts=True)
        stats = dict(zip(unique, counts))

        return stats[self.class_value] / sum(stats.values())
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        return multi_normal_pdf(x, self.mean, self.cov)
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()
    
    

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    
    a = 1 / np.sqrt(2 * np.pi * np.square(std))
    b = np.exp(-1 * np.square(x - mean) / (2 * np.square(std)))

    return a * b
    
def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variante normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """

    a = np.power(2 * np.pi, (-1) * len(mean) / 2) * np.power(np.linalg.det(cov), -0.5)
    b = np.exp(-0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean)))

    return a * b


####################################################################################################
#                                            Part B
####################################################################################################
EPSILLON = 1e-6 # == 0.000001 It could happen that a certain value will only occur in the test set.
                # In case such a thing occur the probability for that value will EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with la place smoothing.
        
        Input
        - dataset: The dataset from which to compute the probabilites (Numpy Array).
        - class_value : Compute the relevant parameters only for instances from the given class.
        """
        self.total_len = len(dataset)
        self.class_value = class_value
        self.class_data = dataset[dataset[:, -1] == class_value][:, :-1]

        # Determines the number of possible values of the relevant attribute
        self.vj = []

        # Determines the number of training instances with the given class
        self.ni = len(self.class_data)

        for col in dataset.T:
            self.vj.append(len(np.unique(col)))

    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.ni / self.total_len
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        result = 1

        # Iterate each attribute to calculate the likelihood.
        for idx, elem in enumerate(x):
            col = self.class_data[:, idx]
            nij = sum(col == x[idx]) if elem in col else EPSILLON
            probability = (nij + 1) / (self.ni + self.vj[idx])

            # Assume independence between attributes.
            result *= probability

        return result
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_prior() * self.get_instance_likelihood(x)

    
####################################################################################################
#                                            General
####################################################################################################            
class MAPClassifier():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a postreiori classifier. 
        This class will hold 2 class distribution, one for class 0 and one for class 1, and will predicit and instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
        
        Input
            - An instance to predict.
            
        Output
            - 0 if the posterior probability of class 0 is higher 1 otherwise.
        """
        posterior0 = self.ccd0.get_instance_posterior(x)
        posterior1 = self.ccd1.get_instance_posterior(x)
        return 0 if posterior0 > posterior1 else 1

    
def compute_accuracy(testset, map_classifier):
    """
    Compute the accuracy of a given a testset and using a map classifier object.
    
    Input
        - testset: The test for which to compute the accuracy (Numpy array).
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / #testset size
    """
    correctly_classified = 0

    for instance in testset:
        # Predict instance classification without including it's classifier value
        result = map_classifier.predict(instance[:-1])
        if result == instance[-1]:
            correctly_classified += 1

    return correctly_classified / len(testset)
