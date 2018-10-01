import random 
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import sys
import json

def initialisation(x, dim):
    """
    x = number of nodes in the linear network 
    dim = the dimension of each weight vector corresponding to each
    nodes in the network  
    """
    net_dim = x
    init_radius = x / 2
    times = n_iterations / np.log(init_radius)
    network = np.random.random((x,dim))
    
    return net_dim, init_radius, times, network 

def decay_radius(initial_radius, i, time_constant):
    return initial_radius * np.exp(-i / time_constant)

def decay_learning_rate(initial_learning_rate, i, n_iterations):
    return initial_learning_rate * np.exp(-i / n_iterations)

def calculate_influence(distance, radius):
    return np.exp(-distance / (2* (radius**2)))

def winning_node(input, network):
    
    best_index = 0
    min_dist = sys.maxsize    
    x, dim = np.shape(network)
    
    for i in range(x):
        dist = (network[i] - input)**2
        dist = np.sum(dist)
        
        if dist < min_dist :
            
            min_dist  = dist
            best_index = i
                
    best = network[best_index]
    
    return best, best_index
            
def som_train(network, data):
    
    index, dim = np.shape(network)
    
    for i in range(n_iterations):        
        
        for t in data :
            t = np.array(t)
            # Find the winning node in the network 
            bmu, bmu_idx = winning_node(t, network)
            
            # decay the SOM parameters
            r = decay_radius(init_radius, i, time_constant)
            l = decay_learning_rate(init_learning_rate, i, n_iterations)
            
            
            for x in range(index):
                    w = network[x]
                    w = np.array(w)
                    # Calculate the absolute value distance 
                    # between the winning node and the each 
                    # other node in the network 
                    w_dist = np.abs(x - bmu_idx)
                    
                    # Find the nodes in the neighbourhood 
                    # of the winning node 
                    
                    if w_dist <= r:
                        # calculate the degree of influence (based on the 1-D distance)
                        influence = calculate_influence(w_dist, r)
                        new_w = w + (l * influence * (t - w))
                        # Update the weight in the network
                        network[x] = new_w
    
    return network

def predict(network, data):
    # Test the trained network with 
    # the data 
    indices = []    
    for t in data :
        t = np.array(t)
        bmu, bmu_idx = winning_node(t, network)
        indices.append(bmu_idx)
        
    return indices
    
def only_string(your_string):
    import re
    output = re.sub(r'\W+', '', your_string)
    return output
    
def get_animals_names():
    f = open('animalnames.txt', 'r')
    x = f.readlines()
    names_list = [only_string(e) for e in x]
    f.close()
    return names_list
    
def test() :
    # Load the data 
    file = np.loadtxt('animals.dat',delimiter=',')
    animal_data = file.reshape(32,84)
    
    # Train the network 
    n_iterations = 20
    init_learning_rate = 0.2
    net_dim, init_radius, time_constant, network  = initialisation(100,84)
    net = som_train(network,animal_data)
    
    # Get the names of each animals
    names_list = get_animals_names()
    index_list = predict(net,animal_data)
    p = zip(names_list, index_list)
    k = list(set(p))
    from collections import OrderedDict
    dd = OrderedDict(sorted(k, key=lambda x: x[1]))
    animals = dict(dd)
    
    f = open('animals_by_resemblance.json', 'w')
    json.dump(animals, f, indent=2)
    f.close()
    
    print(animals)


# test()