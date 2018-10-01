import random 
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import sys

def data_handling(raw_data, norm = True, col = False ):
    
    data = np.copy(raw_data)
    if norm:
        if col:
            # Normalise along each column only
            col_maxes = raw_data.max(axis=0)
            data = raw_data / col_maxes[np.newaxis, :]
        else:
            # Normalise entire dataset 
            # as long linear sequence
            data = raw_data / data.max()
            
    return data

def initialisation(x,y,dim):
    
    net_dim = np.array([x,y])
    init_radius = max(x, y)/2
    times = n_iterations / np.log(init_radius)
    network = np.random.random((x,y,dim))
    
    return net_dim, init_radius, times, network 

def decay_radius(initial_radius, i, time_constant):
    return initial_radius * np.exp(-i / time_constant)

def decay_learning_rate(initial_learning_rate, i, n_iterations):
    return initial_learning_rate * np.exp(-i / n_iterations)

def calculate_influence(distance, radius):
    return np.exp(-distance / (2* (radius**2)))

def winning_node(input, network):
    
    # Initialisation of the initial index 
    #  min_dist and x,y for the shape
    # of the input network map
    
    best_index = [0,0]
    min_dist = sys.maxsize    
    x,y,_ = np.shape(network)
    
    for i in range(x):
        for j in range(y):

            dist = (network[i,j,:] - input)**2
            dist = np.sum(dist)
            
            if dist < min_dist :
                
                min_dist = dist 
                best_index = np.array([i,j])
                
    best = network[best_index[0],best_index[1], :]
    
    return best, best_index
            
def som_train(network):
    
    print(np.shape(network))
    row,col,_ = np.shape(network)
    
    
    for i in range(n_iterations): 
          
        for t in data :
            t = np.array(t) 
            # select a training example at random
            #t = data[:, np.random.randint(0, 2)]

            # find its Best Matching Unit
            bmu, bmu_idx = winning_node(t, network)
            
            # decay the SOM parameters
            r = decay_radius(init_radius, i, time_constant)
            l = decay_learning_rate(init_learning_rate, i, n_iterations)
            
            
            for x in range(row):
                for y in range(col):
                    w = network[x, y, :]
                    # get the 2-D distance (again, not the actual Euclidean distance)
                    w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
                    
                    if w_dist <= r**2:
                        # calculate the degree of influence (based on the 2-D distance)
                        influence = calculate_influence(w_dist, r)
                        new_w = w + (l * influence * (t - w))
                        # commit the new weight
                        network[x, y, :] = new_w
    
    return network
    
def votes() :
    # Load the data 
    file = np.loadtxt('votes.dat',delimiter=',')
    votes_data = file.reshape(349,31)
    
    return votes_data
    
def parties() :
    # Load the data 
    # Coding: 0=no_party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'

    file = np.loadtxt('mpparty.dat')
    d = [int(x) for x in file.tolist()]
    return d
    
def sex() :
    # Load the data 
    # % Coding: Male 0, Female 1
    file = np.loadtxt('mpsex.dat')
    d = [int(x) for x in file.tolist()]
    return d

def district() :
    # Load the data 
    # % Coding: 1 - 29
    file = np.loadtxt('mpdistrict.dat')
    d = [int(x) for x in file.tolist()]
    return d

def get_mixed_data(n = 0):
    
    v = votes()
    p = parties()
    s = sex()
    d = district()
    dico = {}
    if n == 0 : 
        for i in range(349):
            
            temp = p[i]
            if temp not in dico:
                dico[temp] = [v[i]]
            else:
                dico[temp].append([v[i]])
    elif n == 1 :

        for i in range(349):
            
            temp = s[i]
            if temp not in dico:
                dico[temp] = [v[i]]
            else:
                dico[temp].append([v[i]])
    else :

        for i in range(349):
            
            temp = d[i]
            if temp not in dico:
                dico[temp] = [v[i]]
            else:
                dico[temp].append([v[i]])
    
    return dico
           
def result_after_parties(net):
    colormap =[[255,255,0],[2,128,254],[87,185,215],[94,0,0],[255,0,0],[0,252,0],[0,0,252],[188,185,221]]
    colormap = np.array(colormap) / 255

    dic = get_mixed_data(0)
    print("Network Trained ! ")
    
    party_map = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}
    for i in range(8):
        les_parties = dic[i] 
        for x in les_parties :
            # print(x)
            winner, winner_index =  winning_node(x, net)
            party_map[i].append(winner_index.tolist())
        
        
    plot1(net,party_map,colormap)
 


def after_sex(net) :

    colormap =[[0, 0, 254],[249, 90, 118]]
    colormap = np.array(colormap) / 255
    data = votes()
    print(np.shape(data))

    dic = get_mixed_data(1)
    print("Network Trained ! ")
    
    sex_map = {0:[],1:[]}
    for i in range(2):
        les_sex = dic[i] 
        for x in les_sex :
            # print(x)
            winner, winner_index =  winning_node(x, net)
            sex_map[i].append(winner_index.tolist())
        
    plot2(net,sex_map,colormap)
        
        
        
def after_dist(net):
    colormap = np.random.random((29,3))
    data = votes()
    print(np.shape(data))

    dic = get_mixed_data(2)
    print("Network Trained ! ")
    
    district_map = {}
    for i in range(29):
        district_map[i+1] = []
    
    for i in range(29):
        les_sex = dic[i+1] 
        for x in les_sex :
            # print(x)
            winner, winner_index =  winning_node(x, net)
            district_map[i+1].append(winner_index.tolist())
        
        
    plot3(net,district_map,colormap)

def plot1(network,party_map,colormap):
    row,col,_ = np.shape(network)
    fig = plt.figure()
    # setup axes
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim((0, row + 1))
    ax.set_ylim((0, col + 1))
    ax.set_title('Self-Organising Map after %d iterations' % n_iterations)
    
    # plot the rectangles
    for x in range(1, row + 1):
        for y in range(1, col + 1):
            temp = [x,y]
            color = []
            for i in range(8):
                if temp in party_map[i]:
                    color = colormap[i]
                    break
            if len(color) == 0 :
                color = [0,0,0]
            ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,facecolor=color,edgecolor='none'))
    plt.savefig('votes_after_parties.png')
    plt.show()

def plot2(network,sex_map,colormap):
    row,col,_ = np.shape(network)
    fig = plt.figure()
    # setup axes
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim((0, row + 1))
    ax.set_ylim((0, col + 1))
    ax.set_title('Self-Organising Map after %d iterations' % n_iterations)
    
    # plot the rectangles
    for x in range(1, row + 1):
        for y in range(1, col + 1):
            temp = [x,y]
            color = []
            for i in range(2):
                if temp in sex_map[i]:
                    color = colormap[i]
                    break
            if len(color) == 0 :
                color = [0,0,0]
            ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,facecolor=color,edgecolor='none'))
    plt.savefig('votes_after_sex.png')
    plt.show()
    

    
def plot3(network,district_map,colormap):
    row,col,_ = np.shape(network)
    fig = plt.figure()
    # setup axes
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim((0, row + 1))
    ax.set_ylim((0, col + 1))
    ax.set_title('Self-Organising Map after %d iterations' % n_iterations)
    
    # plot the rectangles
    for x in range(1, row + 1):
        for y in range(1, col + 1):
            temp = [x,y]
            color = []
            for i in range(29):
                if temp in district_map[i+1]:
                    color = colormap[i]
                    break
            if len(color) == 0 :
                color = [0,0,0]
            ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,facecolor=color,edgecolor='none'))
    plt.savefig('votes_after_district.png')
    plt.show()
    



data = votes()
n_iterations = 20
init_learning_rate = 0.02
net_dim, init_radius, time_constant, network = initialisation(10,10,31)
net = som_train(network)

result_after_parties(net)
after_sex(net) 
after_dist(net)


# print(votes())
# print("-----")
# print(parties().count(1))
# print()
# print(sex())
# print()
# print(district())
# dic = get_mixed_data()
# print(len(dic)) 
# print("-----")
# print(len(dic[1]))

# n_iterations = 10000
# init_learning_rate = 0.01
# raw_data = np.random.randint(0, 255, (2, 100))
# data = data_handling(raw_data)
# net_dim, init_radius, time_constant, network  = initialisation(5,5,2)
# net = som_train(network)
# plot(net)

