
import numpy as np
import matplotlib.pylab as plt
from RBF_NETWORK import RBFN
from RBFN_DELTA_RULE import RBFN as RBFN2

def sinus_function(x):
    return np.sin(x)
    
def square_sine_function(x):
    temp = np.sin(x)
    return np.where(temp>=0,1,-1)
    
def plot(x,y) :
    #plt.figure()
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('tight')
    plt.show()

def get_sine_data(N = 65):
    
    X = np.linspace(0, 2*np.pi, 65)
    y = sinus_function(2*X)

    X_test = np.linspace(0.05, 2*np.pi, 65)
    y_test = sinus_function(2*X_test)
    
    return X,y, X_test, y_test


def get_square_data(N = 65):
    
    X = np.linspace(0, 2*np.pi, 65)
    y = square_sine_function(2*X)
    X_test = np.linspace(0.05, 2*np.pi, 65)
    y_test = square_sine_function(2*X_test)
    
    return X,y, X_test, y_test

def test_rbf_delta_rule():
    
    n_steps = 65
    X,y, X_test, y_test = get_sine_data()
    
    
    model2 = RBFN2(X, nRBF_units = 10)
    model2.fit(X,y)
    prediction2 = model2.predict(X_test)
    
    c = sum(np.abs(y_test.reshape((n_steps)) -  prediction.reshape((n_steps))))/n_steps
    print("Residual = ", c)
    
    plot(X, y)
    
    plot(X,prediction2)

n_steps = 65
X,y, X_test, y_test = get_sine_data()

model = RBFN(X, nRBF_units = 4)
model.fit(X,y)
prediction = model.predict(X_test)

c = sum(np.abs(y_test.reshape((n_steps)) -  prediction.reshape((n_steps))))/n_steps
print("Residual = ", c)

plot(X, y)
# plot(X, y2)
plot(X,prediction)

