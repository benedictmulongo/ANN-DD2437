
import numpy as np
from generate_data import *
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
#sum(abs(input - prediction_network))/N_samples

class delta_rule(object):
    
    def __init__(self, data, y, eta, epochs=50):
        
        self.data = data
        self.labels = y
        self.eta = eta
        self.iters = epochs
        self.w = self.weights_init(data)
        self.error = 100000
        
    def transform_input(self,data):
        """
        The data X must be presented in the following way :
        
        X = [[x1],[x2],[x3],...,[xn]]
        
        """
        data = np.array(data)
        ndata = len(data)
        bias = np.array([[-1]]*ndata)
        if np.ndim(data) < 2 :
            data = data.reshape((ndata,1))
        X = np.insert(data, [0], bias, axis=1)
        return X
        
    def weights_init(self,data):
        
        nd_w = np.ndim(data) + 1
        weight = np.random.random((nd_w))
        
        return weight
        
    def predict(self,inputs,problem = 'classication'):
        
        X1 = self.transform_input(inputs)
        activations =  np.dot(X1,self.w)
        if problem != 'regression' :
            activations = np.where(activations>0,1,0)

        return activations
        
    def forward(self,inputs, problem = 'classication'):
    
        activations =  np.dot(inputs,self.w)
        if problem != 'regression' :
            activations = np.where(activations>0,1,0)

        return activations
        
    def train_batch(self,inputs,targets):
        """ Train the thing """	
        
        # Add the bias to the data
        X = self.transform_input(inputs)
        error = 0
        epoch = 0
        for n in range(self.iters):
            
            self.activations = self.forward(X)
            self.w -= self.eta*np.dot(np.transpose(X),self.activations-targets)
            error = np.sum((self.activations-targets)**2)/len(inputs)
            print("Epoch ", epoch, " Error = ", error )
            epoch = epoch + 1
    
    
    
class delta_online(object):
    
    def __init__(self, data, y, eta, epochs=50):
        
        self.data = data
        self.labels = y
        self.eta = eta
        self.iters = epochs
        self.w = self.weights_init(data)
        self.error = 100000
        
    def transform_input(self,data):
        """
        The data X must be presented in the following way :
        
        X = [[x1],[x2],[x3],...,[xn]]
        
        """
        data = np.array(data)
        ndata = len(data)
        bias = np.array([[-1]]*ndata)
        if np.ndim(data) < 2 :
            data = data.reshape((ndata,1))
        X = np.insert(data, [0], bias, axis=1)
        return X
        
    def weights_init(self,data):
        
        nd_w = np.ndim(data) + 1
        weight = np.random.random((nd_w))
        
        return weight
        
    def predict(self,inputs,problem = 'classication'):
        
        X1 = self.transform_input(inputs)
        activations =  np.dot(X1,self.w)
        if problem != 'regression' :
            activations = np.where(activations>0,1,0)

        return activations
        
    def forward(self,inputs, problem = 'classication'):
    
        activations =  np.dot(inputs,self.w)
        if problem != 'regression' :
            activations = np.where(activations>0,1,0)

        return activations
        
    def train_online(self,inputs,targets):
        """ Train the thing """	
        
        # Add the bias to the data
        X = self.transform_input(inputs)
        error = 0
        epoch = 0
        for n in range(self.iters):
            error = 0
            for index,  x_i in enumerate(X) : 
                self.activations = self.forward(x_i)
                self.w -= self.eta*np.dot(np.transpose(x_i),self.activations-targets[index])
                error += (self.activations-targets[index])**2
            error = error / len(X)
            print("Epoch ", epoch, " Error = ", error )
            epoch = epoch + 1
            
def normalize(X):
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    return X_std
    

def test_batch() :
    Ax, Ay, Bx, By = gendata(5)
    all_data = np.append(Ax, Bx, axis=0)
    all_labels = np.append(Ay, By, axis=0)
    X = all_data 
    y = all_labels
    
    X = normalize(X)
    ada = delta_rule(X, y, 0.01, 20)
    ada.train_batch(X, y)
    
    plt.figure()
    plot_decision_regions(X, y, clf=ada)
    plt.title('Delta batch')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
def test_online() :
    Ax, Ay, Bx, By = gendata(5)
    all_data = np.append(Ax, Bx, axis=0)
    all_labels = np.append(Ay, By, axis=0)
    X = all_data 
    y = all_labels
    
    X = normalize(X)
    ada = delta_online(X, y, 0.01, 20)
    ada.train_online(X, y)
    
    plt.figure()
    plot_decision_regions(X, y, clf=ada)
    plt.title('Delta online')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


test_batch()
print()
test_online()