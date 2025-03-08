import math
import random 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

rand_seed = 69 #random.randint(0, 100) #9, 13, 23, 47, 49, 50, 69
num_vertices = 100
bases = [3000, 5000, 6000, 6500, 7000, 10000, 12000, 14000, 15000, 17000, 18000, 20000, 25000, 30000, 35000]
# 100: [3000, 5000, 6000, 6500, 7000, 10000, 12000, 14000, 15000, 17000, 18000, 20000, 25000, 30000, 35000]
# 50: [300, 320, 325, 330, 350, 370, 400, 420, 450, 470, 500, 550, 600, 700, 750] 
# 40: [100, 120, 125, 130, 135, 140, 145, 180, 200, 205, 210, 240, 270, 310, 340, 360]
# 30: [40, 50, 60, 65, 70, 71, 72, 73, 80, 90, 100, 105, 110, 120, 130, 150, 160]
# 20: [15, 17, 20, 23, 25, 27, 30, 33, 35]

df = pd.DataFrame(columns = ['rand_seed', 'num_vertices', 'prob_base_num', 'avg_deg'])

for b in bases:

    # Set random seed
    random.seed(rand_seed)

    # Create graph
    G = nx.Graph()
    x_axis = {}
    y_axis = {}
    for i in range(0, num_vertices):
        x_axis[i] = round(random.random(), 3)
        y_axis[i] = round(random.random(), 3)
        G.add_node(i)

    nx.set_node_attributes(G, x_axis, name='x_axis')
    nx.set_node_attributes(G, y_axis, name='y_axis')

    # Create edges between points based on distance and probability
    for i in range(0, num_vertices): # Go from 0 to n-1
        for j in range(i+1, num_vertices): # Go from i+1 to n-1
            u = (G.nodes()[i]['x_axis'], G.nodes()[i]['y_axis'])
            v = (G.nodes()[j]['x_axis'], G.nodes()[j]['y_axis'])

            # Find distance between points
            distance = math.dist(u, v)
            
            # Generate probability if an edge exists
            prob = b**(-distance)
            r = random.random()
            if r < prob: # create an edge
                G.add_edge(i, j)
    
    # Create a table to understand avg degree
    avg_deg = 2 * G.number_of_edges() / G.number_of_nodes()
    df.loc[len(df)] = [rand_seed, num_vertices, b, avg_deg]

df.to_csv("model_one.csv", header=True, index = False)