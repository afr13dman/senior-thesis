# Loop through all graphs and see if i can get number of vertices

# import required module
import os
from gerrychain import Graph
import re
import pandas as pd

# assign directory
directory = 'local copy of data/cnty'
state_vertices = []

# iterate over files in that directory
for state_file in os.listdir(directory):
    f = os.path.join(directory, state_file)
    # check if it is a file
    if os.path.isfile(f):
        # Convert json file to graph object
        state_graph = Graph.from_json(f)
        num_nodes = len(state_graph.nodes())

        state = re.search(r"_.*?\.", state_file)
        map_type = re.search(r"^.*?_", state_file)
        state_vertices.append([state.group()[1:-1], map_type.group()[:-1], num_nodes])
    
df = pd.DataFrame(state_vertices, columns=['State', 'Map Type', 'Number of Vertices'])
print(df)