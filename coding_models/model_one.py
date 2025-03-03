import math
import random 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

rand_seed = 13 #13, 47, 50
num_vertices = 50
bases = [400, 420, 450, 470, 500, 550, 600, 700, 750] # 100, 120, 180, 200, 240, 270, 310, 340, 360, 

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

df.to_csv("coding_models/model_one.csv", header=True, index = False)