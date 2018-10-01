import random 
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import sys
import json

def get_cities():
    f = [0.4, 0.4439, 0.2439, 0.1463, 0.1707, 0.2293, 0.2293, 0.761, 0.5171, 0.9414, 0.8732, 0.6536, 0.6878, 0.5219, 0.8488, 0.3609, 0.6683, 0.2536, 0.6195, 0.2634]
    f  = np.array(f)
    back = f.reshape(10,2)
    # [1,2,3,4,5,6,7,8,9,10]
    return back

def initialisation(x, dim):
    """
    x = number of nodes in the linear network 
    dim = the dimension of each weight vector corresponding to each
    nodes in the network  
    """
    net_dim = x
    init_radius = int(x / 4)
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
                    
                    if w_dist <= r :
                        # calculate the degree of influence (based on the 1-D distance)
                        influence = calculate_influence(w_dist, r)
                        new_w = w + (l * influence * (t - w))
                        # Update the weight in the network
                        network[x] = new_w
                        if bmu_idx == 0 :
                            network[9] = new_w
                        if bmu_idx == 9 :
                            network[0] = new_w
        # Used yo print and save sucessive images
        # of the tour found so far after each iterations
        
        # tour, tour_np = tsp_tour(cities, network)
        # tsp_dist = tsp_distance(tour_np)
        # plot_tour_save(cities, tour,i, round(tsp_dist, 3))
        
    return network

def predict(network, data):
    # Test the trained network with 
    # the data 
    indices = [] 
    tour = np.zeros(10)   
    for i, t in enumerate(data) :
        t = np.array(t)
        bmu, bmu_idx = winning_node(t, network)
        indices.append(bmu_idx)
        tour[i] = bmu_idx + 1
    
    tour[-1] = tour[0]
    print("The tour = ")
    for j, x in enumerate(tour) :
        print(" ",j+1 , " -> ", x )
    print()
    return indices
    
def only_string(your_string):
    import re
    output = re.sub(r'\W+', '', your_string)
    return output
    
def distance(a,b) :
    d = (a - b)**2
    dist = np.sum(d)
    return dist
       
def tsp_tour(cities, network) :
    
    # Find the mapping betweenn each neuron
    # int the network and each city
    city_neurons = {}
    for city_idx, city in enumerate(cities):
        # find nearest neuron
        _,idx = winning_node(city, network)
        if idx not in city_neurons:
            city_neurons[idx] = [city]
        else:
            city_neurons[idx].append(city)

    # order cities according to neuron order
    # This is the order after with the cities should
    # be traversed 
    tsp_order = []
    for neuron_idx in range(len(network)):
        if neuron_idx in city_neurons:
            tsp_order += city_neurons[neuron_idx]

    tour_list = [ x.tolist() for x in tsp_order]
    tour_list.append(tour_list[0])
    return np.array(tour_list), tsp_order

def tsp_distance(tsp_order):

    tsp_distance = distance(tsp_order[0], tsp_order[-1])
    for idx in range(len(tsp_order)-1):
        tsp_distance += distance(tsp_order[idx],tsp_order[idx + 1])

    return tsp_distance

def plot_tour_save(cities, tour, i, dist):

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(cities[:,0],cities[:,1],'.')
    plt.plot(tour[:,0],tour[:,1],'b-')
    b = 'Tour found ( #iteration ' + str(i) + ')'
    d = 'd = '+ str(dist)
    plt.grid(True)
    plt.text(0.1, 1, d)
    plt.title(b)
    plt.xlabel('x')
    plt.ylabel('y')
    #pl.show()
    img = 'tour' + 'iter' + str(i)  +  '.jpeg'
    plt.savefig(img)
    plt.close()


def plot_tour(cities, tour):

    import pylab as pl
    pl.figure()
    plt.grid(True)
    pl.plot(cities[:,0],cities[:,1],'.')
    pl.plot(tour[:,0],tour[:,1],'b-')
    pl.show()
    print()

def test() :
    # Load the data 
    cities = get_cities()
    print(cities)
    # Train the network 
    n_iterations = 20
    init_learning_rate = 0.2
    net_dim, init_radius, time_constant, network  = initialisation(10,2)
    net = som_train(network,cities)
    
    
    print("Hakuna matata = ")
    tour, tour_np = tsp_tour(cities, net)
    plot_tour(cities, tour)
