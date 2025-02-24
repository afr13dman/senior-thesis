import math
import random 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

rand_seed = 13 #13, 47, 50
num_vertices = 20
probs = [19.5]

df = pd.DataFrame(columns = ['rand_seed', 'num_vertices', 'prob_base_num', 'avg_deg'])

for p in probs:

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
            prob = p**(-distance)
            r = random.random()
            if r < prob: # create an edge
                G.add_edge(i, j)
    
    # Creating a table to understand avg degree
    # avg_deg = 2 * G.number_of_edges() / G.number_of_nodes()
    # df.loc[len(df)] = [rand_seed, num_vertices, p, avg_deg]

    # Create a dictionary of node positions to plot a figure
    node_locations = {v: (float(G.nodes()[v]['x_axis']), float(G.nodes()[v]['y_axis'])) for v in G.nodes()}
    nx.draw(G, node_size = 50, pos = node_locations)
    plt.savefig(f'imgs/model_one/{num_vertices}-{rand_seed}-{p}.png')

# df.to_csv("model_one.csv", header=True, index = False)