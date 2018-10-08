import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import json

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



def question1():
    Network = [[-1,-1,1,-1,1,-1,-1,1],
    [-1,-1,-1,-1,-1,1,-1,-1],
    [-1,1,1,-1,-1,1,-1,1]]
    Network = np.array(Network)
    print("Network = ")
    print(Network)
    train_Network = train_batch(Network,True)
    print("Trained Network = ")
    print(train_Network)
    print("is_stable = ", check_network_stability(train_Network,Network))
    # print(set_weights(Network[0]))
    # print(set_weights(Network[1]))
    # print(set_weights(Network[2]))
    x1d = [1,-1,1,-1,1,-1,-1,1]
    x2d = [1,1,-1,-1,-1,1,-1,-1]
    x3d = [1,1,1,-1,1,1,-1,1]
    x1dd = [ 1,  1, -1,  1, -1,  1,  1, 1]
    # synchronous_update(train_Network, x1d) # Converged =  [-1. -1.  1. -1.  1. -1. -1.  1.]
    # synchronous_update(train_Network, x2d)
    # synchronous_update(train_Network, x3d)
    print("Convergence of x1d ")
    t1d = asynchronous_update(train_Network, x1d,random_oder = False, window_size = 8) 
    print("Convergence of x2d ")
    t2d = asynchronous_update(train_Network, x2d,random_oder = False, window_size = 8) 
    print("Convergence of x3d ")
    t3d = asynchronous_update(train_Network, x3d,random_oder = False, window_size = 8)
    print("Convergence of x1d > 1/2 wrong ")
    t1dd = asynchronous_update(train_Network, x1dd,random_oder = False, window_size = 8)
    # asynchronous_update(Network, test1)
    # print("Hamming distance")
    # print(hamming_distance([-1,1,1,-1,1,-1],[1,1,1,-1,-1,-1]))
    
    plot_patterns(Network, 2, 4)
    plot_2patterns(x1d,t1d, 2, 4)
    plot_2patterns(x2d,t2d, 2, 4)
    plot_2patterns(x2d,t3d, 2, 4)
    plot_2patterns(x1dd,t1dd, 2, 4)
    print(colormap())

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
    
    
def load_pictures() :
    # Load the data 
    file = np.loadtxt('pict.dat',delimiter=',')
    mat = file.reshape(11,1024)

    return mat
    
def question2():
    data = load_pictures()
    Network = data[0:3]
    train_Network = train_batch(Network,True)
    print("is_stable = ", check_network_stability(train_Network,Network))
    
    p10 = data[9]
    p1 = data[0]
    # respons = asynchronous_update(train_Network, p10,random_oder = False, window_size = 100) 
    respons = asynchronous_update(train_Network, p10,random_oder = True, window_size = 100) 
    # respons = synchronous_update(tran_Network, p10)
    plot_patterns(data, 32, 32)
    plot_2patterns(p10,respons,32,32)
    
    f = open('inter_states.json')
    mat = json.load(f)
    matrix = np.array(mat['1'])
    print(np.shape(matrix))
    matrix = matrix[[0,5,10,15,20,25],:]
    plot_patternsss(matrix, 32, 32)
    # print(mat['1'][0])
    f.close()
    
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
    
def question3():
    data = load_pictures()
    Network = data[0:3]
    train_Network = train_batch(Network,True)
    print("is_stable = ", check_network_stability(train_Network,Network))
    
    p10 = data[9]
    p1 = data[0]
    
    # 3.1
    # E1 = energy(train_Network, data[0])
    # E2 = energy(train_Network, data[1])
    # E3 = energy(train_Network, data[2])
    # E = [E1,E2,E3]
    # plot_patterns_energy(Network,32,32,E)
    
    # 3.2
    # E1 = energy(train_Network, p10)
    # plot_1patterns_energy(p10,32,32,E1)
    
    # 3.3
    
    # respons = asynchronous_update(train_Network, p10,random_oder = False, window_size = 100) 
    # respons = asynchronous_update(train_Network, p10,random_oder = True, window_size = 500) 
    # respons = synchronous_update(train_Network, p10)
    # plot_patterns(data, 32, 32)
    # plot_2patterns(p10,respons,32,32)
    # 
    # f = open('inter_states_energy.json')
    # mat = json.load(f)
    # matrix = np.array(mat['1'])
    # energy = np.array(mat['2'])
    # energy = [round(e,2) for e in energy]
    # plot_patterns_energy(matrix,32,32,energy)   
    # f.close()
    
    # 3.4
    
    A, test_pattern = create_random_matrix(25)
    print(A)
    print(test_pattern)
    # plot_1patterns(A,4,4)
    # A = make_symetrical(A)
    # train_Network = train_batch(A,True)
    # print("Random W is_stable = ", check_network_stability(train_Network,A))
    respons, E = asynchronous_update(A, test_pattern,random_oder = True, window_size = 5) 
    # plot_2patterns(test_pattern,respons,4,4)
    
    plt.plot(list(range(1,len(E)+1)),E,'g--')
    plt.xlabel('Epoch')
    plt.ylabel('Energy')
    plt.show()
    
    plt.figure()
    A = make_symetrical(A)
    respons, E = asynchronous_update(A, test_pattern,random_oder = True, window_size = 5) 
    plt.plot(list(range(1,len(E)+1)),E,'r--')
    plt.xlabel('Epoch')
    plt.ylabel('Energy')
    plt.show()
   
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
        
    
def question4():
    data = load_pictures()
    Network = data[0:3]
    train_Network = train_batch(Network,True)
    print("is_stable = ", check_network_stability(train_Network,Network))
    
    p1 = data[0]
    p2 = data[1]
    p3 = data[2]
    noise_rate = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    
    # respons = asynchronous_update(train_Network, p10,random_oder = False, window_size = 100) 
    # respons = asynchronous_update(train_Network, p10,random_oder = True, window_size = 500) 
    # respons = synchronous_update(train_Network, p10)
    
    rep_list = []
    for noise in noise_rate :
        p_temp = add_noises(p2,noise)
        respons = synchronous_update(train_Network, p_temp)
        plot_2patterns(p_temp,respons,32,32)
        rep_list.append(respons)
    
    plot_patterns(rep_list, 32, 32)
  
def create_pattern(dim = 4):
    a = -1 
    b = 1

    mu, sigma = 0, 0.1
    # This pattern are biased with 0.3
    # in order to make the data more similar +1 > -1
    # Remove it if not needed
    vec = np.random.normal(mu, sigma, dim) + 0.1
    vec = np.where(vec>0, 1, -1)
    return vec
    
def train_by_index(data, index):
    Network = data[0:index]
    train_Network = train_batch(Network,True)
    print("is_stable = ", check_network_stability(train_Network,Network))
    return train_Network
    
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
    
def question5():
    data = load_pictures()
    # train_Network = train_by_index(data, 4)
    
    # p1 = data[0]
    # p2 = data[1]
    # p3 = data[2]
    # p4 = data[3]
    # p5 = data[4]
    # p6 = data[5]
    # p7 = data[6]
    # noise_rate = [0.2,0.3,0.4,0.5,0.6]
    # 
    # accuracy = tolerance(train_Network,p3)
    # print("Accuracy = ",accuracy )
    acc = []
    for i in range(2,8):
        train_Network = train_by_index(data, i)
        accuracy = tolerance(train_Network,data[i-1])
        print("Iter: ", i, " Accuracy = ", accuracy )
        acc.append(accuracy)
        
    plt.plot(acc,'r--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    
    # print(create_pattern(4))
    # print(create_pattern(4))
    
def question5_random():
    data = []
    for i in range(10):
        temp = create_pattern(1024)
        data.append(temp.tolist())
    data = np.array(data)
    
    acc = []
    for i in range(2,8):
        train_Network = train_by_index(data, i)
        accuracy = tolerance(train_Network,data[i-1])
        print("Iter: ", i, " Accuracy = ", accuracy )
        acc.append(accuracy)
        
    plt.plot(acc,'r--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    
    # print(create_pattern(4))
    # print(create_pattern(4))

def check_network_reliability(Network, list_of_inputs) :
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
    return Stable, count
    


def question51_random():
    data_noises = []
    data_clean = []
    for i in range(300):
        t = create_pattern(100)
        temp = add_noises(t, 0.12 )
        data_clean.append(t.tolist())
        data_noises.append(temp.tolist())
    data_noises = np.array(data_noises)
    data_clean = np.array(data_clean)
    
    stable_patterns_clean = []
    stable_patterns_noises = []
    
    for i in range(1,300):
        train_Network = train_by_index(data_clean, i)
        stable, count = check_network_reliability(train_Network, data_clean[0:i])
        stable_patterns_clean.append(count)
        
        train_Network = train_by_index(data_noises, i)
        stable, count = check_network_reliability(train_Network, data_noises[0:i])
        stable_patterns_noises.append(count)
    plt.figure()
    plt.plot(stable_patterns_clean,'r--')
    plt.xlabel('Pattern')
    plt.ylabel('Stable patterns')
    plt.show()
    
    plt.figure()
    plt.plot(stable_patterns_noises,'b--')
    plt.xlabel('Pattern')
    plt.ylabel('Stable patterns')
    plt.show()


def question52_random():

    data_clean = []
    for i in range(300):
        t = create_pattern(100)
        data_clean.append(t.tolist())


    data_clean = np.array(data_clean)
    
    stable_patterns_clean = []

    
    for i in range(1,300):
        train_Network = train_by_index(data_clean, i)
        stable, count = check_network_reliability(train_Network, data_clean[0:i])
        stable_patterns_clean.append(count)

    plt.figure()
    plt.plot(stable_patterns_clean,'g--')
    plt.xlabel('Pattern')
    plt.ylabel('Stable patterns')
    plt.show()

    
    
question52_random()


# A, test_pattern = create_random_matrix(4)
# print(A)
# print(test_pattern)
# plot_1patterns(A,4,4)

# Epoch:  1  Energy:  0.008078059998472095
# Epoch:  2  Energy:  0.14646030530929344
# Epoch:  3  Energy:  0.22574259291724375
# Epoch:  4  Energy:  -0.31706951898972163
# Epoch:  5  Energy:  -0.31706951898972163