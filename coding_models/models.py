# This file includes functions that will return a graph based on the model chosen 
# given a number of vertices, random seed, and base probability (where applicable)

# Import libraries
import math
import random 
import pandas as pd
import networkx as nx

def create_graph(num_vertices, rand_seed):
    """
    Creates a graph with the given number of vertices at a specified random seed.

    Functions return such graph.
    """
    
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

    # Return the created graph
    return G

# Model 1
def model_one(num_vertices, rand_seed, base):
    """
    Creates a graph where edges are added between points based on probability
    of a base raised to the negative distance between the two points.
    - num_vertices is the number of vertices in the graph
    - rand_seed is the random seed used to generate the points
    - base is the base of the probability

    Function returns the graph.
    """

    # Create graph
    G = create_graph(num_vertices, rand_seed)

    # Add edges between points based on distance and probability
    for i in range(0, num_vertices): # Go from 0 to n-1
        for j in range(i+1, num_vertices): # Go from i+1 to n-1
            u = (G.nodes()[i]['x_axis'], G.nodes()[i]['y_axis'])
            v = (G.nodes()[j]['x_axis'], G.nodes()[j]['y_axis'])

            # Find distance between points
            distance = math.dist(u, v)
            
            # Generate probability if an edge exists
            prob = base**(-distance)
            r = random.random()
            if r < prob: # create an edge
                G.add_edge(i, j)
    
    return G

# Model 2
def model_two(n, rand_seed):
    """
    Creates a graph with only the (5.2n) shortest edges
    where n is the number of vertices in the graph
    and rand_seed is the random seed used to generate the points.

    Function returns the graph.
    """

    # Create graph
    G = create_graph(n, rand_seed)

    # Create a dictionary of all distances between points
    dist_df = pd.DataFrame(columns = ['node1', 'node2', 'distance'])
    for i in range(0, n): # Go from 0 to n-1
        for j in range(i+1, n): # Go from i+1 to n-1
            u = (G.nodes()[i]['x_axis'], G.nodes()[i]['y_axis'])
            v = (G.nodes()[j]['x_axis'], G.nodes()[j]['y_axis'])

            # Find distance between points
            distance = math.dist(u, v)

            # Add it to the dataframe
            dist_df.loc[len(dist_df)] = [i, j, distance]
    
    # Add the (5.2n) shortest edges
    edges = int(5.2 * n)
    dist_sorted = dist_df.sort_values("distance").head(edges)
    for index, row in dist_sorted.iterrows():
        i = row['node1']
        j = row['node2']
        G.add_edge(i, j)
    
    return G
