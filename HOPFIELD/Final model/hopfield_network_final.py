import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import json
"""
Ben mulongo
benedith@kth.se

"""
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

def train_batch(inputs, normalise = False):
    
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
                W[i,j] += inputs[k,i]*inputs[k,j]
                
            
    if normalise :
        W = (1/length)*W - np.diag((nb_patterns/ length)*np.ones(length)) 
    else :
        W = W - np.diag((nb_patterns)*np.ones(length)) 
    
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
    
def is_stable(Network, input):
    stable = False
    act = np.matmul(Network, input)
    activations = np.where(act>0,1,-1)
    if activations.tolist() == input.tolist() :
        stable = True
    return stable 
    
def stable_states(Network, input):
    
    act = np.matmul(Network, input)
    activations = np.where(act>0,1,-1)
    return activations
    
def activations_function(t1, t2):
    
    next_state = np.zeros(len(t2))
    for i, x in enumerate(t2) :
        if x > 0 :
            next_state[i] = 1
        elif x < 0 :
            next_state[i] = -1
        else :
            next_state[i] = t1[i]
            
    return next_state
    
def synchronous_update(Network, input):
    
    t1 = np.array(input) 
    
    while True :

        t2 = np.matmul(Network, t1)
        t3 = activations_function(t1, t2)
        # print("t1 = ", t1)
        # print("t22 = ", t2)
        # print("t33 = ", t3)
        if t3.tolist() == t1.tolist() :
            print("Converged = ", t3)
            print("The energy = ", energy(Network, t1))
            break
        else :
            t1 = t3
     
    return t1
    
def asynchronous_update_once(Network, input, random_oder = True, window_size = 1):
    
    length = np.shape(Network)[0]
    order = np.arange(length)
    # print("Order (1111) = ", order)
    if random_oder or window_size <= 2 :
        np.random.shuffle(order)
    t1 = np.array(input)
    # print("The NETTTT = ", Network)
    # print("The shape = ",np.shape(Network)[0] )
    # print("Order = ", order)
    for i in range(window_size) :
        k = order[i]
        t2 = np.matmul(Network, t1)
        t3 = activations_function(t1, t2)
        t1[k] = t3[k]
        
    return t1
    
def asynchronous_update(Network, input, random_oder = True, window_size = 1):

    t1 = np.array(input)
    count = 0
    Ern = []
    while True :
        count = count + 1
        t3 = asynchronous_update_once(Network, t1, random_oder, window_size)
        
        # print("t1 = ", t1)
        # print("t33 = ", t3)
        E = energy(Network, t3)
        print("Epoch: ", count, " Energy: ", E)
        Ern.append(E)
        if t1.tolist() == t3.tolist():
            print("Asynchronous Convergence = ", t1)
            break
        else :
            t1 = t3
    print("Count = ", count)
    
    return t1, Ern
    
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
    
def make_symetrical(A):
    
    A = np.array(A)
    S = 0.5*(A+ A.T)
    return S
    
    
def load_pictures() :
    # Load the data 
    file = np.loadtxt('pict.dat',delimiter=',')
    mat = file.reshape(11,1024)

    return mat
    
   
def add_noises(pattern, rate = 0.9) :
    
    pattern = np.copy(pattern)
    length = len(pattern)
    order = np.arange(length)
    np.random.shuffle(order)
    
    nb_element = int(np.ceil(length*rate))
    
    for i in range(nb_element) :
        t = pattern[order[i]]
        pattern[order[i]] = -t
        
    return pattern 
    
    
def test1():
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


def test2():
    
    # This is a lsit of 11 pictures of dimension 1024 = 32X32
    data = load_pictures()
    # Take the first three patterns
    Network = data[0:3]
    # Train a hopfield network wiht the three images 
    # Normalize = True 
    train_Network = train_batch(Network,True)
    # Check the network stability for each images 
    print("is_stable = ", check_network_stability(train_Network,Network))
    
    # Get a distorded pattern 
    # p10 is a distorded pattern of p1
    p10 = data[9]
    p1 = data[0]
    """
    Use synchronous update the "little model"
    for p10
    """
    # respons = asynchronous_update(train_Network, p10,random_oder = False, window_size = 100) 
    respons, Ern = asynchronous_update(train_Network, p10,random_oder = True, window_size = 500) 
    # respons = synchronous_update(tran_Network, p10)
    # respons is the restored pattern 
    
    # plot all patterns 
    plot_patterns(data, 32, 32)
    
    # plot a corrupted pattern and its restored pattern 
    plot_2patterns(p10,respons,32,32)
    

test2()
