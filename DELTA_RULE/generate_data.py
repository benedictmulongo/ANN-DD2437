import numpy as np
import random
import numpy.linalg as alg
import math as math
import xlwt
from sklearn.model_selection import train_test_split

def gendata(n = 100):
    
    mA = [4, 4]
    sigmaA = 0.5
    
    mB = [2, -3]
    #mB = [3, -1]
    sigmaB = 0.5
    
    classA_x1 = np.random.randn(1,n)*sigmaA + mA[0]
    classA_x2 = np.random.randn(1,n)*sigmaA + mA[1]
    classA = np.stack((classA_x1, classA_x2), axis=-1)
    classA_labels = n*[0]
    
    
    classB_x1 = np.random.randn(1,n)*sigmaB + mB[0]
    classB_x2 = np.random.randn(1,n)*sigmaB + mB[1]
    classB = np.stack((classB_x1, classB_x2), axis=-1)
    classB_labels = n*[1]
    
    return classA[0], np.array(classA_labels), classB[0], np.array(classB_labels)
    
def gendata_overlap(n = 100):
    
    mA = [3, 4]
    sigmaA = 0.5
    
    #mB = [3, -4]
    mB = [4, 3]
    sigmaB = 0.5
    
    classA_x1 = np.random.randn(1,n)*sigmaA + mA[0]
    classA_x2 = np.random.randn(1,n)*sigmaA + mA[1]
    classA = np.stack((classA_x1, classA_x2), axis=-1)
    classA_labels = n*[0]
    
    classB_x1 = np.random.randn(1,n)*sigmaB + mB[0]
    classB_x2 = np.random.randn(1,n)*sigmaB + mB[1]
    classB = np.stack((classB_x1, classB_x2), axis=-1)
    classB_labels = n*[1]
    
    return classA[0], np.array(classA_labels), classB[0], np.array(classB_labels)

    
def tranform_input(X,y):
    
    nb_data, nb_coord = np.shape(X)
    
    # Add a new colum of -1 in the data matrix
    threshold = [[-1]]*nb_data
    X = np.insert(X, [0], threshold, axis=1)
    
    # Create the weight vectors
    
    w = (nb_coord + 1)*[random.uniform(0,0.2)]
    
    return X,y,w

def shuffle(X, y):
    r = np.random.permutation(len(y))
    return X[r], y[r]
    
def normalize(X):
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    return X_std
    



# X_train, y_train,Ax,Ay, Bx,By= gendata_v3()
# print(len(X_train))
# print(len(y_train))
# print(len(Ax))
# print(len(Ay))
# print(len(Bx))
# print(len(By))
# X, labels = genBlobs(centers=2)
# mu, sigma = mlParams(X,labels)
# plotGaussian(X,labels,mu,sigma)

# mu, sigma = mlParams(Ax,Ay)
# plotGaussian(Ax,Ay,mu,sigma)
# 
# mu, sigma = mlParams(all_data,all_labels)
# plotGaussian(all_data,all_labels,mu,sigma)