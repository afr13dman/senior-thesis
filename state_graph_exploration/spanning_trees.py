'''
lattice graphs for comparison with real data

code taken from lattice_comparison.py
could also look at num_span_trees.py
'''
# Import libraries
import os
import networkx as nx
import numpy as np
from numpy.linalg import slogdet
import matplotlib.pyplot as plt
from gerrychain import Graph
import pandas as pd

# Import model functions
import sys
sys.path.append("coding_models")
import models


#############
# real data #
#############

num_nodes_real = []
tree_const_real = []
log_trees_real = []
q = ['cnty', 't', 'bg']
for p in q:
    directory = "local copy of data/" + p

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # check if it is a file
        if os.path.isfile(file_path):
            print ('opening', filename)
            # Convert json file to graph object
            graph = Graph.from_json(file_path)
            nodes = graph.number_of_nodes()
            
            # calculate graph Laplacian
            print('calculating', filename)
            Lap = nx.laplacian_matrix(graph).toarray()
            T = np.delete(Lap,1,0)
            T = np.delete(T,1,1)
            (sign, logabsdet) = slogdet(T)
            if (sign == 1):
                tree_const_real.append(np.exp(logabsdet/nodes))
                num_nodes_real.append(nodes)
                log_trees_real.append(logabsdet)


#############
#  model 1  #
#############

num_nodes_m1 = []
tree_const_m1 = []
log_trees_m1 = []

# Import data to get rand_seed, num_vertices, and prob base
df = pd.read_excel("coding_models/model_one_results.xlsx", sheet_name="Sheet2", skiprows=2)

for index, row in df.iterrows():
    nodes = int(row["num_vertices"])
    rs = int(row["rand_seed"])
    base = int(row["prob_base_num"])
    G = models.model_one(nodes, rs, base)

    # calculate graph Laplacian
    print('calculating model 1:', nodes, rs, base)
    Lap = nx.laplacian_matrix(graph).toarray()
    T = np.delete(Lap,1,0)
    T = np.delete(T,1,1)
    (sign, logabsdet) = slogdet(T)
    if (sign == 1):
        tree_const_m1.append(np.exp(logabsdet/nodes))
        num_nodes_m1.append(nodes)
        log_trees_m1.append(logabsdet)

#############
#  model 2  #
#############

num_nodes_m2 = []
tree_const_m2 = []
log_trees_m2 = []

# Edit previous imported data to get only rand_seed and num_vertices
new_df = df[["num_vertices", "rand_seed"]].drop_duplicates()

for index, row in new_df.iterrows():
    nodes = int(row["num_vertices"])
    rs = int(row["rand_seed"])
    G = models.model_two(nodes, rs)

    # calculate graph Laplacian
    print('calculating model 2:', nodes, rs)
    Lap = nx.laplacian_matrix(graph).toarray()
    T = np.delete(Lap,1,0)
    T = np.delete(T,1,1)
    (sign, logabsdet) = slogdet(T)
    if (sign == 1):
        tree_const_m2.append(np.exp(logabsdet/nodes))
        num_nodes_m2.append(nodes)
        log_trees_m2.append(logabsdet)


#############
#  model 3  #
#############


###################################
# plot st constant v num of nodes #
###################################

plt.scatter(num_nodes_real, tree_const_real, c=['b']*len(num_nodes_real))
plt.scatter(num_nodes_m1, tree_const_m1, c=['r']*len(num_nodes_m1))
plt.scatter(num_nodes_m2, tree_const_m2, c=['g']*len(num_nodes_m2))
plt.title('ST Constant vs Number of Nodes')
plt.xlabel('Number of Nodes')
plt.ylabel('ST Constant')
plt.legend(['real data','model 1', 'model 2'])
plt.savefig(f'state_graph_exploration/st_cons_real_data_v_models.png')
plt.show()
