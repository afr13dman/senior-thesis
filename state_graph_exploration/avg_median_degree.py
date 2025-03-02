# import required module
import os
from gerrychain import Graph
import re
import pandas as pd
from statistics import median

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

        # Calculate avg degree of graph
        # Prof Cannon's research students had (2 * Edges) / Nodes .... Why?
        avg_degree = 2 * state_graph.number_of_edges() / state_graph.number_of_nodes()

        # Calculate median degree of graph
        degrees = sorted([degree for _, degree in state_graph.degree()], reverse=False)
        median_degree = median(degrees)

        state = re.search(r"_.*?\.", state_file)
        map_type = re.search(r"^.*?_", state_file)
        state_vertices.append([state.group()[1:-1], map_type.group()[:-1], avg_degree, median_degree])
    
df = pd.DataFrame(state_vertices, columns=['State', 'Map Type', 'Avg Degree', 'Median Degree'])
df.to_csv(f"{df_type}_avg_median_deg.csv", header=True, index = False)