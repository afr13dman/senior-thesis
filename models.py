# This file includes functions that will return a graph based on the model chosen 
# given a number of vertices, random seed, and base probability (where applicable)

# Import libraries
import math
import random 
import pandas as pd
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

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
    
    # Check for connected components
    if nx.is_connected(G):
        return G
    else:
        # Find the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        
        # Create a subgraph of the largest connected component
        G_sub = G.subgraph(largest_cc).copy()
        
        return G_sub

# Model 2
def model_two(n, rand_seed):
    """
    Creates a graph with only the (5.4n) shortest edges
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
    
    # Add the (5.4/2)n shortest edges
    edges = int((5.4/2) * n)
    dist_sorted = dist_df.sort_values("distance").head(edges)
    for index, row in dist_sorted.iterrows():
        i = row['node1']
        j = row['node2']
        G.add_edge(i, j)
    
    # Check for connected components
    if nx.is_connected(G):
        return G
    else:
        # Find the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        
        # Create a subgraph of the largest connected component
        G_sub = G.subgraph(largest_cc).copy()
        
        return G_sub

# Model 3
def model_three(n, rand_seed):
    """
    Creates a graph based on Delaunay Triangulation of the vertices.

    Function returns the graph.
    """
    # Create graph
    G = create_graph(n, rand_seed)

    # Create a list of node positions
    points = np.array([[float(G.nodes()[v]['x_axis']), float(G.nodes()[v]['y_axis'])] for v in G.nodes()])

    # Compute the Delaunay triangulation and build the graph
    tri = Delaunay(points)
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                G.add_edge(simplex[i], simplex[j])

    return G

# Model 4
def model_dt_with_removal(n, rand_seed, remove_prob=0.2):
    """
    Create a graph based on Delaunay Triangulation, then randomly remove edges.

    Function returns the graph.
    """
    G = create_graph(n, rand_seed)

    # Create a list of node positions
    points = np.array([[float(G.nodes()[v]['x_axis']), float(G.nodes()[v]['y_axis'])] for v in G.nodes()])

    # Compute Delaunay triangulation and build the graph
    tri = Delaunay(points)
    edges_to_remove = []
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (simplex[i], simplex[j])
                if random.random() < remove_prob:
                    edges_to_remove.append(edge)
                else:
                    G.add_edge(*edge)

    # Remove edges
    G.remove_edges_from(edges_to_remove)

    # Check for connected components
    if nx.is_connected(G):
        return G
    else:
        # Find the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        
        # Create a subgraph of the largest connected component
        G_sub = G.subgraph(largest_cc).copy()
        
        return G_sub

# Model 5
def model_dt_with_removal_and_add(n, rand_seed, remove_prob=0.2, add_prob=0.05):
    """
    Create a graph based on Delaunay Triangulation.
    Then randomly remove edges and randomly add edges.

    Function returns the graph.
    """
    G = create_graph(n, rand_seed)

    # Create a list of node positions
    points = np.array([[float(G.nodes()[v]['x_axis']), float(G.nodes()[v]['y_axis'])] for v in G.nodes()])

    # Compute Delaunay triangulation and build the graph
    tri = Delaunay(points)
    edges_to_remove = []
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (simplex[i], simplex[j])
                if random.random() < remove_prob:
                    edges_to_remove.append(edge)
                else:
                    G.add_edge(*edge)

    # Remove edges
    G.remove_edges_from(edges_to_remove)

    # Try adding new edges
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    for _ in range(int(add_prob * num_nodes * (num_nodes - 1) / 2)):  # Limit number of new edges
        u, v = random.sample(nodes, 2)  # Pick two random nodes
        if not G.has_edge(u, v):  # Ensure edge doesn't already exist
            distance = math.dist((G.nodes[u]['x_axis'], G.nodes[u]['y_axis']),
                                 (G.nodes[v]['x_axis'], G.nodes[v]['y_axis']))
            if random.random() < (1.5 ** -distance):  # Distance-based probability
                G.add_edge(u, v)

    # Check for connected components
    if nx.is_connected(G):
        return G
    else:
        # Find the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        
        # Create a subgraph of the largest connected component
        G_sub = G.subgraph(largest_cc).copy()
        
        return G_sub

# Model 6
def model_dt_add_short_edges(n, rand_seed):
    """
    Creates a graph based on Delaunay Triangulation of the vertices.
    Then adds some of the shortest edges.

    Function returns the graph.
    """
    # Create graph
    G = create_graph(n, rand_seed)

    # Create a list of node positions
    points = np.array([[float(G.nodes()[v]['x_axis']), float(G.nodes()[v]['y_axis'])] for v in G.nodes()])

    # Compute the Delaunay triangulation and build the graph
    tri = Delaunay(points)
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                G.add_edge(simplex[i], simplex[j])

    # Create a dictionary of all distances between points
    dist_df = pd.DataFrame(columns=['node1', 'node2', 'distance'])
    for i in range(0, n):
        for j in range(i + 1, n):
            u = (G.nodes[i]['x_axis'], G.nodes[i]['y_axis'])
            v = (G.nodes[j]['x_axis'], G.nodes[j]['y_axis'])
            distance = math.dist(u, v)
            dist_df.loc[len(dist_df)] = [i, j, distance]
    
    # Add the next n shortest edges that aren't yet in the graph
    edges = n
    dist_sorted = dist_df.sort_values("distance")
    for index, row in dist_sorted.iterrows():
        i = row['node1']
        j = row['node2']
        if not G.has_edge(i, j):
            G.add_edge(i, j)
            edges -= 1
        if edges == 0:
            return G
    
    return G

# Model 7
def model_dt_add_short_remove_long(n, rand_seed, remove_fraction=0.2):
    """
    Creates a graph based on Delaunay Triangulation of the vertices,
    adds some of the shortest edges and removes a fraction of the longest ones.

    Function returns the modified graph.
    """
    G = model_dt_add_short_edges(n, rand_seed)  # Start with Model 6

    # Collect all edges with distances
    edge_list = [(u, v, math.dist((G.nodes[u]['x_axis'], G.nodes[u]['y_axis']),
                                  (G.nodes[v]['x_axis'], G.nodes[v]['y_axis']))) 
                 for u, v in G.edges()]
    
    # Sort by distance (longest first)
    edge_list.sort(key=lambda x: x[2], reverse=True)

    # Remove a fraction of longest edges
    num_to_remove = int(len(edge_list) * remove_fraction)
    edges_to_remove = edge_list[:num_to_remove]
    G.remove_edges_from([(u, v) for u, v, _ in edges_to_remove])

    # Check for connected components
    if nx.is_connected(G):
        return G
    else:
        # Find the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        
        # Create a subgraph of the largest connected component
        G_sub = G.subgraph(largest_cc).copy()
        
        return G_sub

# Model 8
def model_eight(n, rand_seed, scaling_factor=0.5, remove_prob=0.6):
    """
    Creates a graph based on Delaunay Triangulation of the vertices,
    add edges via preferential attachment.

    Function returns the graph.
    """
    G = create_graph(n, rand_seed)

    # Create a list of node positions
    points = np.array([[float(G.nodes()[v]['x_axis']), float(G.nodes()[v]['y_axis'])] for v in G.nodes()])

    # Compute the Delaunay triangulation and build the graph
    tri = Delaunay(points)
    edges_to_remove = []
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (simplex[i], simplex[j])
                if random.random() < remove_prob:
                    edges_to_remove.append(edge)
                else:
                    G.add_edge(*edge)

    # Remove edges
    G.remove_edges_from(edges_to_remove)
                
    # Define number of new edges based on scaling factor
    num_new_edges = int(scaling_factor * n)

    # Adds edges via preferential attachment 
    degrees = np.array([G.degree(n) for n in G.nodes()])
    node_list = list(G.nodes())
    
    for _ in range(num_new_edges):
        # Select a random node
        node1 = random.choice(node_list)
        
        # Compute preferential probabilities
        degree_sum = degrees.sum()
        probabilities = degrees / degree_sum if degree_sum > 0 else np.ones(len(degrees)) / len(degrees)
        
        # Select another node with probability proportional to its degree
        node2 = np.random.choice(node_list, p=probabilities)
        
        # Avoid self-loops and duplicate edges
        if node1 != node2 and not G.has_edge(node1, node2):
            G.add_edge(node1, node2)
            
            # Update degrees array
            degrees[node_list.index(node1)] += 1
            degrees[node_list.index(node2)] += 1
    
    # Check for connected components
    if nx.is_connected(G):
        return G
    else:
        # Find the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        
        # Create a subgraph of the largest connected component
        G_sub = G.subgraph(largest_cc).copy()
        
        return G_sub


# Model 9
def model_dt_with_removal_add_shortest_edges(n, rand_seed, remove_prob=0.2):
    """
    Create a graph based on Delaunay Triangulation, then randomly remove edges,
    next add shortest edges.

    Function returns the graph.
    """
    G = create_graph(n, rand_seed)

    # Create a list of node positions
    points = np.array([[float(G.nodes()[v]['x_axis']), float(G.nodes()[v]['y_axis'])] for v in G.nodes()])

    # Compute Delaunay triangulation and build the graph
    tri = Delaunay(points)
    edges_to_remove = []
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (simplex[i], simplex[j])
                if random.random() < remove_prob:
                    edges_to_remove.append(edge)
                else:
                    G.add_edge(*edge)

    # Remove edges
    G.remove_edges_from(edges_to_remove)
    
    # Create a dictionary of all distances between points
    dist_df = pd.DataFrame(columns=['node1', 'node2', 'distance'])
    for i in range(0, n):
        for j in range(i + 1, n):
            u = (G.nodes[i]['x_axis'], G.nodes[i]['y_axis'])
            v = (G.nodes[j]['x_axis'], G.nodes[j]['y_axis'])
            distance = math.dist(u, v)
            dist_df.loc[len(dist_df)] = [i, j, distance]
    
    # Add the next n shortest edges that aren't yet in the graph
    edges = n
    dist_sorted = dist_df.sort_values("distance")
    for index, row in dist_sorted.iterrows():
        i = row['node1']
        j = row['node2']
        if not G.has_edge(i, j):
            G.add_edge(i, j)
            edges -= 1
        if edges == 0:
            # Check for connected components
            if nx.is_connected(G):
                return G
            else:
                # Find the largest connected component
                largest_cc = max(nx.connected_components(G), key=len)
                
                # Create a subgraph of the largest connected component
                G_sub = G.subgraph(largest_cc).copy()
                return G_sub


# Model 10
def model_dt_add_shortest_edges_remove_rand(n, rand_seed, remove_prob=0.2):
    """
    Create a graph based on Delaunay Triangulation, add shortest edges, then randomly remove edges.

    Function returns the graph.
    """
    G = create_graph(n, rand_seed)

    # Create a list of node positions
    points = np.array([[float(G.nodes()[v]['x_axis']), float(G.nodes()[v]['y_axis'])] for v in G.nodes()])

    # Compute Delaunay triangulation and build the graph
    tri = Delaunay(points)
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                G.add_edge(simplex[i], simplex[j])

    # Create a dictionary of all distances between points
    dist_df = pd.DataFrame(columns=['node1', 'node2', 'distance'])
    for i in range(0, n):
        for j in range(i + 1, n):
            u = (G.nodes[i]['x_axis'], G.nodes[i]['y_axis'])
            v = (G.nodes[j]['x_axis'], G.nodes[j]['y_axis'])
            distance = math.dist(u, v)
            dist_df.loc[len(dist_df)] = [i, j, distance]
    
    # Add the next n shortest edges that aren't yet in the graph
    edges = n
    dist_sorted = dist_df.sort_values("distance")
    for index, row in dist_sorted.iterrows():
        i = row['node1']
        j = row['node2']
        if not G.has_edge(i, j):
            G.add_edge(i, j)
            edges -= 1
        if edges == 0:
            # Remove Edges    
            edges_to_remove = []
            for edge in G.edges():
                if random.random() < remove_prob:
                    edges_to_remove.append(edge)

            G.remove_edges_from(edges_to_remove)

            # Check for connected components
            if nx.is_connected(G):
                return G
            else:
                # Find the largest connected component
                largest_cc = max(nx.connected_components(G), key=len)
                
                # Create a subgraph of the largest connected component
                G_sub = G.subgraph(largest_cc).copy()
                return G_sub


# Model 11
def model_two_rand_removal(n, rand_seed, scaling_factor=6.8, remove_prob=0.2):
    """
    Creates a graph with only the (5.4n) shortest edges
    where n is the number of vertices in the graph
    and rand_seed is the random seed used to generate the points.
    Then randomly remove points

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
    
    # for model 11 scaling_factor 6.8, removal 0.2
    # for model 11b scaling_factor 9, removal 0.4
    # for model 11c scaling_factor 14, removal 0.6
    edges = int((scaling_factor/2) * n) 
    dist_sorted = dist_df.sort_values("distance").head(edges)
    for index, row in dist_sorted.iterrows():
        i = row['node1']
        j = row['node2']
        G.add_edge(i, j)
    
    # Remove Edges    
    edges_to_remove = []
    for edge in G.edges():
        if random.random() < remove_prob:
            edges_to_remove.append(edge)

    G.remove_edges_from(edges_to_remove)
    
    # Check for connected components
    if nx.is_connected(G):
        return G
    else:
        # Find the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        
        # Create a subgraph of the largest connected component
        G_sub = G.subgraph(largest_cc).copy()
        
        return G_sub