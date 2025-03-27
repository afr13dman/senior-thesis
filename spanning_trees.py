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
import pickle

# Import model functions
import models


#############
# functions #
#############

def check_file_paths(file_paths):
    """
    Takes a list of file paths and checks if the exists.

    Returns True if they all exist, False otherwise.
    """
    files_exist = True

    for file_path in file_paths:
        if not os.path.exists(file_path):
            files_exist = False
    
    return files_exist


#############
# real data #
#############

file_paths = ['pkls/num_nodes_real.pkl', 'pkls/tree_const_real.pkl', 'pkls/log_trees_real.pkl']
if check_file_paths(file_paths):
    with open('pkls/num_nodes_real.pkl', 'rb') as f:
        num_nodes_real = pickle.load(f)
    with open('pkls/tree_const_real.pkl', 'rb') as f:
        tree_const_real = pickle.load(f)
    with open('pkls/log_trees_real.pkl', 'rb') as f:
        log_trees_real = pickle.load(f)
else:
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
                    tree_const_real.append(logabsdet/nodes)
                    num_nodes_real.append(nodes)
                    log_trees_real.append(logabsdet)

    # Save the lists as pickles so I do not have to run the code each time
    with open('pkls/num_nodes_real.pkl', 'wb') as f:
        pickle.dump(num_nodes_real, f)
    with open('pkls/tree_const_real.pkl', 'wb') as f:
        pickle.dump(tree_const_real, f)
    with open('pkls/log_trees_real.pkl', 'wb') as f:
        pickle.dump(log_trees_real, f)

#################
#  Import Data  #
#################

# Import data to get rand_seed, num_vertices, and prob base
df = pd.read_excel("model_one_results.xlsx", sheet_name="5.4", skiprows=2)
new_df = df[["num_vertices", "rand_seed"]].drop_duplicates()

new_vert = [200, 400, 600] #, 800, 10000, 12000, 14000, 16000, 18000, 20000, 25000]
for vert in new_vert:
    for seed in df["rand_seed"].unique()[0:2]:
        new_row = pd.DataFrame([{'num_vertices': vert, 'rand_seed': seed}])
        new_df = pd.concat([new_df, new_row], ignore_index=True)

#############
#  model 2  #
#############

file_paths = ['pkls/num_nodes_m2.pkl', 'pkls/tree_const_m2.pkl', 'pkls/log_trees_m2.pkl']
if check_file_paths(file_paths):
    with open('pkls/num_nodes_m2.pkl', 'rb') as f:
        num_nodes_m2 = pickle.load(f)
    with open('pkls/tree_const_m2.pkl', 'rb') as f:
        tree_const_m2 = pickle.load(f)
    with open('pkls/log_trees_m2.pkl', 'rb') as f:
        log_trees_m2 = pickle.load(f)
else:
    num_nodes_m2 = []
    tree_const_m2 = []
    log_trees_m2 = []

    for index, row in new_df.iterrows():
        nodes = int(row["num_vertices"])
        rs = int(row["rand_seed"])
        graph = models.model_two(nodes, rs)

        # calculate graph Laplacian
        print('calculating model 2:', nodes, rs)
        Lap = nx.laplacian_matrix(graph).toarray()
        T = np.delete(Lap,1,0)
        T = np.delete(T,1,1)
        (sign, logabsdet) = slogdet(T)
        if (sign == 1):
            tree_const_m2.append(logabsdet/nodes)
            num_nodes_m2.append(nodes)
            log_trees_m2.append(logabsdet)

    # Save the lists as pickles so I do not have to run the code each time
    with open('pkls/num_nodes_m2.pkl', 'wb') as f:
        pickle.dump(num_nodes_m2, f)
    with open('pkls/tree_const_m2.pkl', 'wb') as f:
        pickle.dump(tree_const_m2, f)
    with open('pkls/log_trees_m2.pkl', 'wb') as f:
        pickle.dump(log_trees_m2, f)


#############
#  model 3  #
#############

file_paths = ['pkls/num_nodes_m3.pkl', 'pkls/tree_const_m3.pkl', 'pkls/log_trees_m3.pkl']
if check_file_paths(file_paths):
    with open('pkls/num_nodes_m3.pkl', 'rb') as f:
        num_nodes_m3 = pickle.load(f)
    with open('pkls/tree_const_m3.pkl', 'rb') as f:
        tree_const_m3 = pickle.load(f)
    with open('pkls/log_trees_m3.pkl', 'rb') as f:
        log_trees_m3 = pickle.load(f)
else:
    num_nodes_m3 = []
    tree_const_m3 = []
    log_trees_m3 = []

    for index, row in new_df.iterrows():
        nodes = int(row["num_vertices"])
        rs = int(row["rand_seed"])
        graph = models.model_three(nodes, rs)

        # calculate graph Laplacian
        print('calculating model 3:', nodes, rs)
        Lap = nx.laplacian_matrix(graph).toarray()
        T = np.delete(Lap,1,0)
        T = np.delete(T,1,1)
        (sign, logabsdet) = slogdet(T)
        if (sign == 1):
            tree_const_m3.append(np.exp(logabsdet/nodes))
            num_nodes_m3.append(nodes)
            log_trees_m3.append(logabsdet)
    
    # Save the lists as pickles so I do not have to run the code each time
    with open('pkls/num_nodes_m3.pkl', 'wb') as f:
        pickle.dump(num_nodes_m3, f)
    with open('pkls/tree_const_m3.pkl', 'wb') as f:
        pickle.dump(tree_const_m3, f)
    with open('pkls/log_trees_m3.pkl', 'wb') as f:
        pickle.dump(log_trees_m3, f)


#############
#  model 4  #
#############

file_paths = ['pkls/num_nodes_m4.pkl', 'pkls/tree_const_m4.pkl', 'pkls/log_trees_m4.pkl']
if check_file_paths(file_paths):
    with open('pkls/num_nodes_m4.pkl', 'rb') as f:
        num_nodes_m4 = pickle.load(f)
    with open('pkls/tree_const_m4.pkl', 'rb') as f:
        tree_const_m4 = pickle.load(f)
    with open('pkls/log_trees_m4.pkl', 'rb') as f:
        log_trees_m4 = pickle.load(f)
else:
    num_nodes_m4 = []
    tree_const_m4 = []
    log_trees_m4 = []

    for index, row in new_df.iterrows():
        nodes = int(row["num_vertices"])
        rs = int(row["rand_seed"])
        graph = models.model_three_with_removal(nodes, rs)

        # calculate graph Laplacian
        print('calculating model 4:', nodes, rs)
        Lap = nx.laplacian_matrix(graph).toarray()
        T = np.delete(Lap,1,0)
        T = np.delete(T,1,1)
        (sign, logabsdet) = slogdet(T)
        if (sign == 1):
            tree_const_m4.append(np.exp(logabsdet/nodes))
            num_nodes_m4.append(nodes)
            log_trees_m4.append(logabsdet)
    
    # Save the lists as pickles so I do not have to run the code each time
    with open('pkls/num_nodes_m4.pkl', 'wb') as f:
        pickle.dump(num_nodes_m4, f)
    with open('pkls/tree_const_m4.pkl', 'wb') as f:
        pickle.dump(tree_const_m4, f)
    with open('pkls/log_trees_m4.pkl', 'wb') as f:
        pickle.dump(log_trees_m4, f)

##############
#  model 4b  #
##############

# Model 4 but choosing remove prob as 0.4 instead of default 0.2
# Remove prob 0.5 made st constant a bit lower than real data
# Remove prob 0.3 made st constant too high
remove_prob = 0.4 # Seems best probability regarding spanning tree constant close to real data

file_paths = ['pkls/num_nodes_m4b.pkl', 'pkls/tree_const_m4b.pkl', 'pkls/log_trees_m4b.pkl']
if check_file_paths(file_paths):
    with open('pkls/num_nodes_m4b.pkl', 'rb') as f:
        num_nodes_m4b = pickle.load(f)
    with open('pkls/tree_const_m4b.pkl', 'rb') as f:
        tree_const_m4b = pickle.load(f)
    with open('pkls/log_trees_m4b.pkl', 'rb') as f:
        log_trees_m4b = pickle.load(f)
else:
    num_nodes_m4b = []
    tree_const_m4b = []
    log_trees_m4b = []

    for index, row in new_df.iterrows():
        nodes = int(row["num_vertices"])
        rs = int(row["rand_seed"])
        graph = models.model_three_with_removal(nodes, rs, remove_prob)

        # calculate graph Laplacian
        print('calculating model 4b:', nodes, rs)
        Lap = nx.laplacian_matrix(graph).toarray()
        T = np.delete(Lap,1,0)
        T = np.delete(T,1,1)
        (sign, logabsdet) = slogdet(T)
        if (sign == 1):
            tree_const_m4b.append(np.exp(logabsdet/nodes))
            num_nodes_m4b.append(nodes)
            log_trees_m4b.append(logabsdet)
    
    # Save the lists as pickles so I do not have to run the code each time
    with open('pkls/num_nodes_m4b.pkl', 'wb') as f:
        pickle.dump(num_nodes_m4b, f)
    with open('pkls/tree_const_m4b.pkl', 'wb') as f:
        pickle.dump(tree_const_m4b, f)
    with open('pkls/log_trees_m4b.pkl', 'wb') as f:
        pickle.dump(log_trees_m4b, f)
    
###################################
# plot st constant v num of nodes #
###################################

# PLOT AGAINST EACH OTHER

plt.scatter(num_nodes_real, tree_const_real, c=['b']*len(num_nodes_real))
plt.scatter(num_nodes_m1, tree_const_m1, c=['r']*len(num_nodes_m1))
plt.scatter(num_nodes_m2, tree_const_m2, c=['g']*len(num_nodes_m2))
plt.scatter(num_nodes_m3, tree_const_m3, c=['y']*len(num_nodes_m3))
plt.scatter(num_nodes_m4, tree_const_m4, c=['m']*len(num_nodes_m4))
plt.scatter(num_nodes_m4b, tree_const_m4b, c=['c']*len(num_nodes_m4b))
plt.title('ST Constant vs Number of Nodes for Real Data')
plt.xlabel('Number of Nodes')
plt.ylabel('ST Constant')
plt.legend(['real data','model 1', 'model 2', 'model 3', 'model 4', 'model 4b'])
plt.savefig(f'imgs/st_cons/st_cons_real_data_vs_models.png')
plt.show()

# Zoom in
plt.scatter(num_nodes_real, tree_const_real, c=['b']*len(num_nodes_real))
plt.scatter(num_nodes_m1, tree_const_m1, c=['r']*len(num_nodes_m1))
plt.scatter(num_nodes_m2, tree_const_m2, c=['g']*len(num_nodes_m2))
plt.scatter(num_nodes_m3, tree_const_m3, c=['y']*len(num_nodes_m3))
plt.scatter(num_nodes_m4, tree_const_m4, c=['m']*len(num_nodes_m4))
plt.scatter(num_nodes_m4b, tree_const_m4b, c=['c']*len(num_nodes_m4b))
plt.title('ST Constant vs Number of Nodes for Real Data')
plt.xlabel('Number of Nodes')
plt.ylabel('ST Constant')
plt.xlim(0, 1000)
plt.legend(['real data','model 1', 'model 2', 'model 3', 'model 4', 'model 4b'])
plt.savefig(f'imgs/st_cons/st_cons_zoomed_in_real_data_vs_models.png')
plt.show()

# PLOT REAL DATA AND MODELS SEPARATELY 
# plt.scatter(num_nodes_real, tree_const_real, c=['b']*len(num_nodes_real))
# plt.title('ST Constant vs Number of Nodes for Real Data')
# plt.xlabel('Number of Nodes')
# plt.ylabel('ST Constant')
# plt.savefig(f'imgs/st_cons/st_cons_real_data.png')
# plt.show()

# plt.scatter(num_nodes_m1, tree_const_m1, c=['r']*len(num_nodes_m1))
# plt.title('ST Constant vs Number of Nodes for Model 1')
# plt.xlabel('Number of Nodes')
# plt.ylabel('ST Constant')
# plt.savefig(f'imgs/st_cons/st_cons_models1.png')
# plt.show()

# plt.scatter(num_nodes_m2, tree_const_m2, c=['g']*len(num_nodes_m2))
# plt.title('ST Constant vs Number of Nodes for Model 2')
# plt.xlabel('Number of Nodes')
# plt.ylabel('ST Constant')
# plt.savefig(f'imgs/st_cons/st_cons_models2.png')
# plt.show()

# plt.scatter(num_nodes_m3, tree_const_m3, c=['y']*len(num_nodes_m3))
# plt.title('ST Constant vs Number of Nodes for Model 3')
# plt.xlabel('Number of Nodes')
# plt.ylabel('ST Constant')
# plt.savefig(f'imgs/st_cons/st_cons_models3.png')
# plt.show()