import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from delta_rule import delta_online

class RBFN(object):

    def __init__(self, inputs, nRBF_units, sigma = 0):

        if sigma==0:
            # Set width of Gaussians
            d = (inputs.max(axis=0)-inputs.min(axis=0)).max()
            self.sigma = d/np.sqrt(2*nRBF_units)  
        else:
            self.sigma = sigma
        input_shape = np.ndim(inputs)
        self.input_shape = input_shape
        self.nRBF_units = nRBF_units
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_point):
        return np.exp(-self.sigma*np.linalg.norm(center-data_point)**2)

    def _calculate_interpolation_matrix(self,X):
        """ Calculates interpolation matrix G using self._kernel_function
        # Arguments
            X: Training data
        # Input shape
            (num_data_samples, input_shape)
        # Returns
            G: Interpolation matrix
        """
        G = np.zeros((X.shape[0], self.nRBF_units))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg,center_arg] = self._kernel_function(center,
                                                                data_point)
        return G

    def fit(self,X,Y):

        # Get a random indices in the range of 0 - X.length
        random_args = np.random.permutation(X.shape[0]).tolist()
        # Get the corresponding random data folloing indices 
        # random_args
        random_data = [X[arg] for arg in random_args]
        # Initialize the nRBF_units hidden layer Weight from
        # random data 
        # get all data from 0 to #nRBF_units
        self.centers = random_data[:self.nRBF_units]
        
        # G is the hidden layer no bias is used
        G = self._calculate_interpolation_matrix(X)
        # calculate the weights the hidden layer 
        # with LEAST Square pseudo inverse 
        self.weights = np.dot(np.linalg.pinv(G),Y)
        # Calculate the weights with delta rule 

        self.Delta_rule = delta_online(G, 0.05, 50)
        self.Delta_rule.train_online(G,Y)

    def predict(self,X):
        """
        # Arguments
            X: test data
        # Input shape
            (num_test_samples, input_shape)
        """
        G = self._calculate_interpolation_matrix(X)
        ret = self.Delta_rule.predict(G,problem = 'regression')

        return ret

    def normalize(self,X):
        X_std = np.copy(X)
        X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
        X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
        return X_std