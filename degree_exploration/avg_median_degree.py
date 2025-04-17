# import required module
import os
from gerrychain import Graph
import re
import pandas as pd
from statistics import median
import networkx as nx

def max_degree(G):
    # Initialize max_degree
    max_degree = -1

    # Iterate over all nodes and their degrees
    for degree in G.degree():
        if degree[1] > max_degree:
            max_degree = degree[1]
    
    return max_degree

# assign directory
df_type = 'bg' # Work with either tracts (t) or block groups (bg)
directory = 'local copy of data/' + df_type + '/'
state_vertices = []

# iterate over files in that directory
for state_file in os.listdir(directory):
    f = os.path.join(directory, state_file)

    # check if it is a file
    if os.path.isfile(f):
        # Convert json file to graph object
        state_graph = Graph.from_json(f)

        # Planar?
        planar = nx.check_planarity(state_graph)[0]
        
        # Connected?
        connected = nx.is_connected(state_graph)

        # Calculate avg degree of graph
        avg_degree = 2 * state_graph.number_of_edges() / state_graph.number_of_nodes()

        # Calculate median degree of graph
        degrees = sorted([degree for _, degree in state_graph.degree()], reverse=False)
        median_degree = median(degrees)

        # Calculate max degree
        max_deg = max_degree(state_graph)

        state = re.search(r"_.*?\.", state_file)
        map_type = re.search(r"^.*?_", state_file)
        state_vertices.append([state.group()[1:-1], map_type.group()[:-1], 
                               planar, connected,
                               avg_degree, median_degree, max_deg])
    
df = pd.DataFrame(state_vertices, columns=['State', 'Map Type', 'Planar', 'Connected',
                                            'Avg Degree', 'Median Degree', 'Max Degree'])
df.to_csv(f"degree_exploration/{df_type}_avg_median_deg.csv", header=True, index = False)