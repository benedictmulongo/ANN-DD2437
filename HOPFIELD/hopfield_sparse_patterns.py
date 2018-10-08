import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import json
from random import *

def colormap(n = 3) :
    seq1 = ['viridis', 'plasma', 'inferno', 'magma']
    seq2 = ['Greys', 'Purples', 'Blues', 'Greens', 'PuBuGn', 'BuGn', 'YlGn']
    div = ['PiYG', 'PRGn', 'bwr', 'seismic']
    misc = ['flag', 'prism', 'ocean',  'nipy_spectral', 'gist_ncar']
    
    map =[seq1,seq2,div,misc]
    color = map[n]
    length = len(color)
    order = np.arange(length)
    np.random.shuffle(order)
    return color[order[0]]

def hamming_distance(x,y):
    
    d = 0
    if len(x) == len(y):
        d = np.abs(np.array(x) - np.array(y))
        d = np.sum(d)
        d = d / 2
    else :
        d = 1000000
    
    return d
            
def complement(x):
    x = np.array(x)
    return -1*x

def train_batch(inputs, rho = 0.1, normalise = False):
    
    """
    The inputs should be presented like this:
    
    X = [[a1,a2,a3,..., an], 0
         [b1,b2,b3,..., bn], 1
         ...                 ...
         [z1,z2,z3,..., zn]] k
         
         n -> length of the pattern's sequence 
         k -> number of patterns 
    """
    nb_patterns, length = np.shape(inputs)
    W = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            for k in range(nb_patterns):
                W[i,j] += (inputs[k,i]-rho)*(inputs[k,j]-rho)
                if i == j :
                    W[i,j] = 0
                
            
    if normalise :
        W = (1/length)*W 
    else :
        W = W 
    
    return W
    
def train_sequential(W, input,normalise = False):
    nb_patterns, length = np.shape(W)
    result = W + set_weights(input)
    if normalise :
        result = (1/length)*result 
    
    return result 
    
def set_weights(input):
    
    dimension = len(input)

    W = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            
            W[i,j] = input[i]*input[j]

    W = W - np.diag(np.ones(dimension)) 
    
    return W
    
def check_network_stability(Network, list_of_inputs) :
    """
    Given a list of input 
    X = [[X1], [X2], ...,[Xn]] 
    and a hopfield network N
    check the network stability given
    te inputs 

    """
    count = 0
    input_index = []
    Stable = False 
    for i, input in enumerate(list_of_inputs) :
        
        if is_stable(Network, input):
            count = count + 1
        else :
            input_index.append(i)
    
    if count != len(list_of_inputs):
        print("The network is instable for the input indices = ", input_index)
    else :
        Stable = True
        print("The network is stable ! congratulations ")
    return Stable
    
def is_stable(Network, input,  theta):
    stable = False    
    act = np.matmul(Network, input)
    act = act - theta
    activations = 0.5 + 0.5*np.sign(act)
    
    if activations.tolist() == input.tolist() :
        stable = True
    # print("Input = ", input)
    # print("Activation = ",activations )
    return stable 
    
def stable_states(Network, input, theta):
    
    act = np.matmul(Network, input)
    act = act - theta
    activations = 0.5 + 0.5*np.sign(act)
    
    return activations
    

    
def energy(Network, input):
    result = np.matmul(Network, input)
    E = -np.dot(result, input)
    return E

def plot_patterns(list_of_patterns, row, col) :
    
    nb_patterns = len(list_of_patterns)
    fig, ax = plt.subplots(1, nb_patterns, figsize=(13, 5))
    k = ['bwr', 'seismic']
    for i in range(nb_patterns):
        ax[i].matshow(list_of_patterns[i].reshape((row, col)), cmap=k[0])
        ax[i].set_title('Pattern ' + str(i))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        
    plt.show()
    
def plot_2patterns(A1,A2, row, col) :
    
    A1 = np.array(A1)
    A2 = np.array(A2)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].matshow(A1.reshape(row, col), cmap='bwr')
    ax[0].set_title('Corrupted pattern')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    
    ax[1].matshow(A2.reshape(row, col), cmap='bwr')
    ax[1].set_title('Recovered pattern')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    
    plt.show()
    
def plot_1patterns(A, row, col) :
    
    # Display matrix
    A = np.array(A)
    plt.matshow(A.reshape(row, col), cmap='bwr')
    plt.title('Pattern')
    plt.xticks([])
    plt.yticks([])

    plt.show()
    
def plot_patternsss(list_of_patterns, row, col) :
    
    nb_patterns = len(list_of_patterns)
    fig, ax = plt.subplots(1, nb_patterns, figsize=(13, 5))
    k = ['bwr', 'seismic']
    for i in range(nb_patterns):
        ax[i].matshow(list_of_patterns[i].reshape((row, col)), cmap=k[0])
        ax[i].set_title('Epoch ' + str(i*5))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        
    plt.show()
    
def test():
    # X = [[1,-1,1,-1,1,1],[1,1,1,-1,-1,-1]]
    # X = np.array(X)
    # # 
    # print(train_batch(X,True))
    # print()
    # print(set_weights(X[0]))
    # 
    # print("Recall = ", X[0])
    # recall = stable_states(train_batch(X,True), X[0])
    # print("Respons = ", recall)
    # print("is_stable = ", check_network_stability(train_batch(X,True),X))
    
    # print()
    # t1 = [1,-1,1,1,-1]
    # t2 = [0.4,0,0.4,-0.8,0]
    # print(activations_function(t1, t2))
    
    
    
    # Network = [[0,-0.2,0.2,-0.2,-0.2],[-0.2,0,-0.2,0.2,0.2],[0.2,-0.2,0,-0.2,-0.2],[-0.2,0.2,-0.2,0,0.2],[-0.2,0.2,-0.2,0.2,0]]
    # Network = np.array(Network)
    # print("Network = ")
    # print(Network)
    # synchronous_update(Network, [1,1,1,1,-1])
    # asynchronous_update(Network, [1,1,1,1,-1])
    
    # Network = [[0,-2,2,0],[-2,0,-2,0],[2,-2,0,0],[0,0,0,0]]
    # Network = np.array(Network)
    # print("Network = ")
    # print(Network)
    # test1 = [1,1,-1,1]
    # test2 = [-1,-1,1,1]
    # synchronous_update(Network, test1)
    # asynchronous_update(Network, test1)
    print("Hamming distance")
    print(hamming_distance([-1,1,1,-1,1,-1],[1,1,1,-1,-1,-1]))


def create_random_matrix(dim = 4):
    a = -1 
    b = 1

    mu, sigma = 0, 0.1
    vec = np.random.normal(mu, sigma, dim)
    mat = np.random.normal(mu, sigma,(dim,dim)) 
    # mat = np.where(mat>0, 1, -1)
    vec = np.where(vec>0, 1, -1)
    return mat, vec
    
def make_symetrical(A):
    
    A = np.array(A)
    S = 0.5*(A+ A.T)
    return S
    
    
def plot_patterns_energy(list_of_patterns, row, col, E_list) :
    
    nb_patterns = len(list_of_patterns)
    fig, ax = plt.subplots(1, nb_patterns, figsize=(13, 5))
    k = ['bwr', 'seismic']
    for i in range(nb_patterns):
        ax[i].matshow(list_of_patterns[i].reshape((row, col)), cmap=k[0])
        ax[i].set_title('E = ' + str(E_list[i]))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        
    plt.show()
    
def plot_1patterns_energy(A, row, col,E) :
    
    # Display matrix
    A = np.array(A)
    plt.matshow(A.reshape(row, col), cmap='bwr')
    plt.title('Energy = ' + str(E))
    plt.xticks([])
    plt.yticks([])

    plt.show()
    
        
def create_sparse_pattern(dim = 4, N=5, rho=0.1):

    
    data = []
    for i in range(N):
        number  = np.random.random(dim)
        vec = np.where(number<rho, 1, 0)
        data.append(vec.tolist())
        
    return data
    
def tolerance(train_Network,pattern):
    rep_list = []
    noise_rate = [0.2,0.3,0.4,0.5,0.6]
    accuracy = 0
    count = 0
    for noise in noise_rate :
        count = count + 1
        p_temp = add_noises(pattern,noise)
        respons = synchronous_update(train_Network, p_temp)
        # plot_2patterns(p_temp,respons,32,32)
        rep_list.append(respons)
        if pattern.tolist() == respons.tolist() :
            accuracy = accuracy + 1
    # plot_patterns(rep_list, 32, 32)
    
    success = accuracy / count
    return success


def check_network_reliability(Network, list_of_inputs, theta) :
    """
    Given a list of input 
    X = [[X1], [X2], ...,[Xn]] 
    and a hopfield network N
    check the network stability given
    te inputs 

    """
    count = 0
    input_index = []
    Stable = False 
    for i, input in enumerate(list_of_inputs) :
        
        if is_stable(Network, input,  theta):
            count = count + 1
        else :
            input_index.append(i)
    
    if count != len(list_of_inputs):
        print("The network is instable for the input indices = ", input_index)
    else :
        Stable = True
        print("The network is stable ! congratulations ")
    return Stable, count
    
    
def train_by_index(data, index, rho):
    Network = data[0:index]
    train_Network = train_batch(Network,rho, False)
    # print("is_stable = ", check_network_stability(train_Network,Network))
    return train_Network
    
def sparsity() :
    dim = 100
    N = 100
    rho = [0.1,0.05,0.01]
    theta = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.84, 0.99]
    data = [np.array(create_sparse_pattern(dim, N, rho=rho[0])),
            np.array(create_sparse_pattern(dim, N, rho=rho[1])),
            np.array(create_sparse_pattern(dim, N, rho=rho[2]))]
             
    # print(" Data 1 0.1")
    # print(data[0])
    # train_Network = train_batch(data[0],rho[0], True)
    # print("Trained network ")
    # print(train_Network)
    # respons = stable_states(train_Network, data[0][2],theta[1])
    # print(data[0][2])
    # print()
    # print(respons)
    # print(is_stable(train_Network, data[0][2],  theta[1]))
    
    stable_patterns = []
    for i in range(1,100):
        train_Network = train_by_index(data[0], i, rho[0])
        stable, count = check_network_reliability(train_Network, data[0][0:i-1],theta[8])
        stable_patterns.append(count)

    #   plt.figure()
    plt.plot(stable_patterns,'g--')
    plt.xlabel('Pattern')
    plt.ylabel('Stable patterns')
    plt.show()
    
def sparsity1() :
    dim = 100
    N = 100
    rho = [0.1,0.05,0.01]
    theta = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.84, 0.99]
    data = [np.array(create_sparse_pattern(dim, N, rho=rho[0])),
            np.array(create_sparse_pattern(dim, N, rho=rho[1])),
            np.array(create_sparse_pattern(dim, N, rho=rho[2]))]
             
    # print(" Data 1 0.1")
    # print(data[0])
    # train_Network = train_batch(data[0],rho[0], True)
    # print("Trained network ")
    # print(train_Network)
    # respons = stable_states(train_Network, data[0][2],theta[1])
    # print(data[0][2])
    # print()
    # print(respons)
    # print(is_stable(train_Network, data[0][2],  theta[1]))
    
    stable_patterns = []
    for i in range(1,100):
        train_Network = train_by_index(data[2], i, rho[2])
        stable, count = check_network_reliability(train_Network, data[2][0:i-1],theta[8])
        stable_patterns.append(count)

    #   plt.figure()
    plt.plot(stable_patterns,'g--')
    plt.xlabel('Pattern')
    plt.ylabel('Stable patterns')
    plt.show()

sparsity1() 