'''
code taken and edited from lattice_comparison.py
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
import importlib
import models
importlib.reload(models)

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


#############
#  model 1  #
#############

# Import data to get rand_seed, num_vertices, and prob base
df = pd.read_csv("model_one_desired_avg_deg.csv")

file_paths = ['pkls/num_nodes_m1.pkl', 'pkls/tree_const_m1.pkl', 'pkls/log_trees_m1.pkl']
if check_file_paths(file_paths):
    with open('pkls/num_nodes_m1.pkl', 'rb') as f:
        num_nodes_m1 = pickle.load(f)
    with open('pkls/tree_const_m1.pkl', 'rb') as f:
        tree_const_m1 = pickle.load(f)
    with open('pkls/log_trees_m1.pkl', 'rb') as f:
        log_trees_m1 = pickle.load(f)
else:
    num_nodes_m1 = []
    tree_const_m1 = []
    log_trees_m1 = []

    for index, row in df.iterrows():
        nodes = int(row["num_vertices"])
        rs = int(row["rand_seed"])
        base = int(row["prob_base_num"])
        graph = models.model_one(nodes, rs, base)

        # calculate graph Laplacian
        print('calculating model 1:', nodes, rs, base)
        Lap = nx.laplacian_matrix(graph).toarray()
        T = np.delete(Lap,1,0)
        T = np.delete(T,1,1)
        (sign, logabsdet) = slogdet(T)
        if (sign == 1):
            tree_const_m1.append(logabsdet/nodes)
            num_nodes_m1.append(nodes)
            log_trees_m1.append(logabsdet)

    # Save the lists as pickles so I do not have to run the code each time
    with open('pkls/num_nodes_m1.pkl', 'wb') as f:
        pickle.dump(num_nodes_m1, f)
    with open('pkls/tree_const_m1.pkl', 'wb') as f:
        pickle.dump(tree_const_m1, f)
    with open('pkls/log_trees_m1.pkl', 'wb') as f:
        pickle.dump(log_trees_m1, f)


###############
# Update Data #
###############

# Edit previous imported data to get only rand_seed and num_vertices
new_df = df[["num_vertices", "rand_seed"]].drop_duplicates()

new_vert = [200, 400] #, 600, 800] #, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 5000, 7000, 10000] #, 15000, 20000, 25000]
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
        graph = models.model_dt_with_removal(nodes, rs)

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
        graph = models.model_dt_with_removal(nodes, rs, remove_prob)

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


#############
#  model 5  #
#############

file_paths = ['pkls/num_nodes_m5.pkl', 'pkls/tree_const_m5.pkl', 'pkls/log_trees_m5.pkl']
if check_file_paths(file_paths):
    with open('pkls/num_nodes_m5.pkl', 'rb') as f:
        num_nodes_m5 = pickle.load(f)
    with open('pkls/tree_const_m5.pkl', 'rb') as f:
        tree_const_m5 = pickle.load(f)
    with open('pkls/log_trees_m5.pkl', 'rb') as f:
        log_trees_m5 = pickle.load(f)
else:
    num_nodes_m5 = []
    tree_const_m5 = []
    log_trees_m5 = []

    for index, row in new_df.iterrows():
        nodes = int(row["num_vertices"])
        rs = int(row["rand_seed"])
        graph = models.model_dt_with_removal_and_add(nodes, rs)

        # calculate graph Laplacian
        print('calculating model 5:', nodes, rs)
        Lap = nx.laplacian_matrix(graph).toarray()
        T = np.delete(Lap,1,0)
        T = np.delete(T,1,1)
        (sign, logabsdet) = slogdet(T)
        if (sign == 1):
            tree_const_m5.append(logabsdet/nodes)
            num_nodes_m5.append(nodes)
            log_trees_m5.append(logabsdet)

    # Save the lists as pickles so I do not have to run the code each time
    with open('pkls/num_nodes_m5.pkl', 'wb') as f:
        pickle.dump(num_nodes_m5, f)
    with open('pkls/tree_const_m5.pkl', 'wb') as f:
        pickle.dump(tree_const_m5, f)
    with open('pkls/log_trees_m5.pkl', 'wb') as f:
        pickle.dump(log_trees_m5, f)


##############
#  model 5b  #
##############

file_paths = ['pkls/num_nodes_m5b.pkl', 'pkls/tree_const_m5b.pkl', 'pkls/log_trees_m5b.pkl']
if check_file_paths(file_paths):
    with open('pkls/num_nodes_m5b.pkl', 'rb') as f:
        num_nodes_m5b = pickle.load(f)
    with open('pkls/tree_const_m5b.pkl', 'rb') as f:
        tree_const_m5b = pickle.load(f)
    with open('pkls/log_trees_m5b.pkl', 'rb') as f:
        log_trees_m5b = pickle.load(f)
else:
    num_nodes_m5b = []
    tree_const_m5b = []
    log_trees_m5b = []

    for index, row in new_df.iterrows():
        nodes = int(row["num_vertices"])
        rs = int(row["rand_seed"])
        graph = models.model_dt_with_removal_and_add(nodes, rs, remove_prob=0.4)

        # calculate graph Laplacian
        print('calculating model 5b:', nodes, rs)
        Lap = nx.laplacian_matrix(graph).toarray()
        T = np.delete(Lap,1,0)
        T = np.delete(T,1,1)
        (sign, logabsdet) = slogdet(T)
        if (sign == 1):
            tree_const_m5b.append(logabsdet/nodes)
            num_nodes_m5b.append(nodes)
            log_trees_m5b.append(logabsdet)

    # Save the lists as pickles so I do not have to run the code each time
    with open('pkls/num_nodes_m5b.pkl', 'wb') as f:
        pickle.dump(num_nodes_m5b, f)
    with open('pkls/tree_const_m5b.pkl', 'wb') as f:
        pickle.dump(tree_const_m5b, f)
    with open('pkls/log_trees_m5b.pkl', 'wb') as f:
        pickle.dump(log_trees_m5b, f)

#############
#  model 6  #
#############

file_paths = ['pkls/num_nodes_m6.pkl', 'pkls/tree_const_m6.pkl', 'pkls/log_trees_m6.pkl']
if check_file_paths(file_paths):
    with open('pkls/num_nodes_m6.pkl', 'rb') as f:
        num_nodes_m6 = pickle.load(f)
    with open('pkls/tree_const_m6.pkl', 'rb') as f:
        tree_const_m6 = pickle.load(f)
    with open('pkls/log_trees_m6.pkl', 'rb') as f:
        log_trees_m6 = pickle.load(f)
else:
    num_nodes_m6 = []
    tree_const_m6 = []
    log_trees_m6 = []

    for index, row in new_df.iterrows():
        nodes = int(row["num_vertices"])
        rs = int(row["rand_seed"])
        graph = models.model_dt_add_short_edges(nodes, rs)

        # calculate graph Laplacian
        print('calculating model 6:', nodes, rs)
        Lap = nx.laplacian_matrix(graph).toarray()
        T = np.delete(Lap,1,0)
        T = np.delete(T,1,1)
        (sign, logabsdet) = slogdet(T)
        if (sign == 1):
            tree_const_m6.append(np.exp(logabsdet/nodes))
            num_nodes_m6.append(nodes)
            log_trees_m6.append(logabsdet)
    
    # Save the lists as pickles so I do not have to run the code each time
    with open('pkls/num_nodes_m6.pkl', 'wb') as f:
        pickle.dump(num_nodes_m6, f)
    with open('pkls/tree_const_m6.pkl', 'wb') as f:
        pickle.dump(tree_const_m6, f)
    with open('pkls/log_trees_m6.pkl', 'wb') as f:
        pickle.dump(log_trees_m6, f)


#############
#  model 7  #
#############

file_paths = ['pkls/num_nodes_m7.pkl', 'pkls/tree_const_m7.pkl', 'pkls/log_trees_m7.pkl']
if check_file_paths(file_paths):
    with open('pkls/num_nodes_m7.pkl', 'rb') as f:
        num_nodes_m7 = pickle.load(f)
    with open('pkls/tree_const_m7.pkl', 'rb') as f:
        tree_const_m7 = pickle.load(f)
    with open('pkls/log_trees_m7.pkl', 'rb') as f:
        log_trees_m7 = pickle.load(f)
else:
    num_nodes_m7 = []
    tree_const_m7 = []
    log_trees_m7 = []

    for index, row in new_df.iterrows():
        nodes = int(row["num_vertices"])
        rs = int(row["rand_seed"])
        graph = models.model_dt_add_short_remove_long(nodes, rs)

        # calculate graph Laplacian
        print('calculating model 7:', nodes, rs)
        Lap = nx.laplacian_matrix(graph).toarray()
        T = np.delete(Lap,1,0)
        T = np.delete(T,1,1)
        (sign, logabsdet) = slogdet(T)
        if (sign == 1):
            tree_const_m7.append(np.exp(logabsdet/nodes))
            num_nodes_m7.append(nodes)
            log_trees_m7.append(logabsdet)
    
    # Save the lists as pickles so I do not have to run the code each time
    with open('pkls/num_nodes_m7.pkl', 'wb') as f:
        pickle.dump(num_nodes_m7, f)
    with open('pkls/tree_const_m7.pkl', 'wb') as f:
        pickle.dump(tree_const_m7, f)
    with open('pkls/log_trees_m7.pkl', 'wb') as f:
        pickle.dump(log_trees_m7, f)


#############
#  model 8  #
#############

file_paths = ['pkls/num_nodes_m8.pkl', 'pkls/tree_const_m8.pkl', 'pkls/log_trees_m8.pkl']
if check_file_paths(file_paths):
    with open('pkls/num_nodes_m8.pkl', 'rb') as f:
        num_nodes_m8 = pickle.load(f)
    with open('pkls/tree_const_m8.pkl', 'rb') as f:
        tree_const_m8 = pickle.load(f)
    with open('pkls/log_trees_m8.pkl', 'rb') as f:
        log_trees_m8 = pickle.load(f)
else:
    num_nodes_m8 = []
    tree_const_m8 = []
    log_trees_m8 = []

    for index, row in new_df.iterrows():
        nodes = int(row["num_vertices"])
        rs = int(row["rand_seed"])
        graph = models.model_eight(nodes, rs)

        # calculate graph Laplacian
        print('calculating model 8:', nodes, rs)
        Lap = nx.laplacian_matrix(graph).toarray()
        T = np.delete(Lap,1,0)
        T = np.delete(T,1,1)
        (sign, logabsdet) = slogdet(T)
        if (sign == 1):
            tree_const_m8.append(np.exp(logabsdet/nodes))
            num_nodes_m8.append(nodes)
            log_trees_m8.append(logabsdet)
    
    # Save the lists as pickles so I do not have to run the code each time
    with open('pkls/num_nodes_m8.pkl', 'wb') as f:
        pickle.dump(num_nodes_m8, f)
    with open('pkls/tree_const_m8.pkl', 'wb') as f:
        pickle.dump(tree_const_m8, f)
    with open('pkls/log_trees_m8.pkl', 'wb') as f:
        pickle.dump(log_trees_m8, f)

# ##############
# #  model 8b  #
# ##############

# file_paths = ['pkls/num_nodes_m8b.pkl', 'pkls/tree_const_m8b.pkl', 'pkls/log_trees_m8b.pkl']
# if check_file_paths(file_paths):
#     with open('pkls/num_nodes_m8b.pkl', 'rb') as f:
#         num_nodes_m8b = pickle.load(f)
#     with open('pkls/tree_const_m8b.pkl', 'rb') as f:
#         tree_const_m8b = pickle.load(f)
#     with open('pkls/log_trees_m8b.pkl', 'rb') as f:
#         log_trees_m8b = pickle.load(f)
# else:
#     num_nodes_m8b = []
#     tree_const_m8b = []
#     log_trees_m8b = []

#     for index, row in new_df.iterrows():
#         nodes = int(row["num_vertices"])
#         rs = int(row["rand_seed"])
#         graph = models.model_eight(nodes, rs) # try different removal prob and scaling factors

#         # calculate graph Laplacian
#         print('calculating model 8b:', nodes, rs)
#         Lap = nx.laplacian_matrix(graph).toarray()
#         T = np.delete(Lap,1,0)
#         T = np.delete(T,1,1)
#         (sign, logabsdet) = slogdet(T)
#         if (sign == 1):
#             tree_const_m8b.append(np.exp(logabsdet/nodes))
#             num_nodes_m8b.append(nodes)
#             log_trees_m8b.append(logabsdet)
    
#     # Save the lists as pickles so I do not have to run the code each time
#     with open('pkls/num_nodes_m8b.pkl', 'wb') as f:
#         pickle.dump(num_nodes_m8b, f)
#     with open('pkls/tree_const_m8b.pkl', 'wb') as f:
#         pickle.dump(tree_const_m8b, f)
#     with open('pkls/log_trees_m8b.pkl', 'wb') as f:
#         pickle.dump(log_trees_m8b, f)


#############
#  model 9  #
#############

file_paths = ['pkls/num_nodes_m9.pkl', 'pkls/tree_const_m9.pkl', 'pkls/log_trees_m9.pkl']
if check_file_paths(file_paths):
    with open('pkls/num_nodes_m9.pkl', 'rb') as f:
        num_nodes_m9 = pickle.load(f)
    with open('pkls/tree_const_m9.pkl', 'rb') as f:
        tree_const_m9 = pickle.load(f)
    with open('pkls/log_trees_m9.pkl', 'rb') as f:
        log_trees_m9 = pickle.load(f)
else:
    num_nodes_m9 = []
    tree_const_m9 = []
    log_trees_m9 = []

    for index, row in new_df.iterrows():
        nodes = int(row["num_vertices"])
        rs = int(row["rand_seed"])
        graph = models.model_dt_with_removal_add_shortest_edges(nodes, rs)

        # calculate graph Laplacian
        print('calculating model 9:', nodes, rs)
        Lap = nx.laplacian_matrix(graph).toarray()
        T = np.delete(Lap,1,0)
        T = np.delete(T,1,1)
        (sign, logabsdet) = slogdet(T)
        if (sign == 1):
            tree_const_m9.append(np.exp(logabsdet/nodes))
            num_nodes_m9.append(nodes)
            log_trees_m9.append(logabsdet)
    
    # Save the lists as pickles so I do not have to run the code each time
    with open('pkls/num_nodes_m9.pkl', 'wb') as f:
        pickle.dump(num_nodes_m9, f)
    with open('pkls/tree_const_m9.pkl', 'wb') as f:
        pickle.dump(tree_const_m9, f)
    with open('pkls/log_trees_m9.pkl', 'wb') as f:
        pickle.dump(log_trees_m9, f)

##############
#  model 9b  #
##############

file_paths = ['pkls/num_nodes_m9b.pkl', 'pkls/tree_const_m9b.pkl', 'pkls/log_trees_m9b.pkl']
if check_file_paths(file_paths):
    with open('pkls/num_nodes_m9b.pkl', 'rb') as f:
        num_nodes_m9b = pickle.load(f)
    with open('pkls/tree_const_m9b.pkl', 'rb') as f:
        tree_const_m9b = pickle.load(f)
    with open('pkls/log_trees_m9b.pkl', 'rb') as f:
        log_trees_m9b = pickle.load(f)
else:
    num_nodes_m9b = []
    tree_const_m9b = []
    log_trees_m9b = []

    for index, row in new_df.iterrows():
        nodes = int(row["num_vertices"])
        rs = int(row["rand_seed"])
        graph = models.model_dt_with_removal_add_shortest_edges(nodes, rs, remove_prob=0.4)

        # calculate graph Laplacian
        print('calculating model 9b:', nodes, rs)
        Lap = nx.laplacian_matrix(graph).toarray()
        T = np.delete(Lap,1,0)
        T = np.delete(T,1,1)
        (sign, logabsdet) = slogdet(T)
        if (sign == 1):
            tree_const_m9b.append(np.exp(logabsdet/nodes))
            num_nodes_m9b.append(nodes)
            log_trees_m9b.append(logabsdet)
    
    # Save the lists as pickles so I do not have to run the code each time
    with open('pkls/num_nodes_m9b.pkl', 'wb') as f:
        pickle.dump(num_nodes_m9b, f)
    with open('pkls/tree_const_m9b.pkl', 'wb') as f:
        pickle.dump(tree_const_m9b, f)
    with open('pkls/log_trees_m9b.pkl', 'wb') as f:
        pickle.dump(log_trees_m9b, f)


##############
#  model 10  #
##############

file_paths = ['pkls/num_nodes_m10.pkl', 'pkls/tree_const_m10.pkl', 'pkls/log_trees_m10.pkl']
if check_file_paths(file_paths):
    with open('pkls/num_nodes_m10.pkl', 'rb') as f:
        num_nodes_m10 = pickle.load(f)
    with open('pkls/tree_const_m10.pkl', 'rb') as f:
        tree_const_m10 = pickle.load(f)
    with open('pkls/log_trees_m10.pkl', 'rb') as f:
        log_trees_m10 = pickle.load(f)
else:
    num_nodes_m10 = []
    tree_const_m10 = []
    log_trees_m10 = []

    for index, row in new_df.iterrows():
        nodes = int(row["num_vertices"])
        rs = int(row["rand_seed"])
        graph = models.model_dt_add_shortest_edges_remove_rand(nodes, rs)

        # calculate graph Laplacian
        print('calculating model 10:', nodes, rs)
        Lap = nx.laplacian_matrix(graph).toarray()
        T = np.delete(Lap,1,0)
        T = np.delete(T,1,1)
        (sign, logabsdet) = slogdet(T)
        if (sign == 1):
            tree_const_m10.append(np.exp(logabsdet/nodes))
            num_nodes_m10.append(nodes)
            log_trees_m10.append(logabsdet)
    
    # Save the lists as pickles so I do not have to run the code each time
    with open('pkls/num_nodes_m10.pkl', 'wb') as f:
        pickle.dump(num_nodes_m10, f)
    with open('pkls/tree_const_m10.pkl', 'wb') as f:
        pickle.dump(tree_const_m10, f)
    with open('pkls/log_trees_m10.pkl', 'wb') as f:
        pickle.dump(log_trees_m10, f)

###############
#  model 10b  #
###############

file_paths = ['pkls/num_nodes_m10b.pkl', 'pkls/tree_const_m10b.pkl', 'pkls/log_trees_m10b.pkl']
if check_file_paths(file_paths):
    with open('pkls/num_nodes_m10b.pkl', 'rb') as f:
        num_nodes_m10b = pickle.load(f)
    with open('pkls/tree_const_m10b.pkl', 'rb') as f:
        tree_const_m10b = pickle.load(f)
    with open('pkls/log_trees_m10b.pkl', 'rb') as f:
        log_trees_m10b = pickle.load(f)
else:
    num_nodes_m10b = []
    tree_const_m10b = []
    log_trees_m10b = []

    for index, row in new_df.iterrows():
        nodes = int(row["num_vertices"])
        rs = int(row["rand_seed"])
        graph = models.model_dt_add_shortest_edges_remove_rand(nodes, rs, remove_prob=0.4)

        # calculate graph Laplacian
        print('calculating model 10b:', nodes, rs)
        Lap = nx.laplacian_matrix(graph).toarray()
        T = np.delete(Lap,1,0)
        T = np.delete(T,1,1)
        (sign, logabsdet) = slogdet(T)
        if (sign == 1):
            tree_const_m10b.append(np.exp(logabsdet/nodes))
            num_nodes_m10b.append(nodes)
            log_trees_m10b.append(logabsdet)
    
    # Save the lists as pickles so I do not have to run the code each time
    with open('pkls/num_nodes_m10b.pkl', 'wb') as f:
        pickle.dump(num_nodes_m10b, f)
    with open('pkls/tree_const_m10b.pkl', 'wb') as f:
        pickle.dump(tree_const_m10b, f)
    with open('pkls/log_trees_m10b.pkl', 'wb') as f:
        pickle.dump(log_trees_m10b, f)

###################################
# plot st constant v num of nodes #
###################################

# PLOT AGAINST EACH OTHER

plt.scatter(num_nodes_real, tree_const_real, c=['b']*len(num_nodes_real))
plt.scatter(num_nodes_m6, tree_const_m6, c=['tab:orange']*len(num_nodes_m6))
plt.scatter(num_nodes_m7, tree_const_m7, c=['tab:brown']*len(num_nodes_m7))
plt.scatter(num_nodes_m8, tree_const_m8, c=['g']*len(num_nodes_m8))
#plt.scatter(num_nodes_m8b, tree_const_m8b, c=['y']*len(num_nodes_m8b))
plt.scatter(num_nodes_m9, tree_const_m9, c=['tab:red']*len(num_nodes_m9))
plt.scatter(num_nodes_m9b, tree_const_m9b, c=['tab:pink']*len(num_nodes_m9b))
plt.scatter(num_nodes_m10, tree_const_m10, c=['m']*len(num_nodes_m10))
plt.scatter(num_nodes_m10b, tree_const_m10b, c=['tab:purple']*len(num_nodes_m10b))
plt.title('ST Constant vs Number of Nodes for Real Data')
plt.xlabel('Number of Nodes')
plt.ylabel('ST Constant')
plt.legend(['real data', 'model 6', 'model 7', 'model 8', 'model 9', 'model 9b', 'model 10', 'model 10b'])
plt.savefig(f'imgs/st_cons/st_cons_real_data_vs_models.png')
plt.show()

# Zoom in
plt.scatter(num_nodes_real, tree_const_real, c=['b']*len(num_nodes_real))
plt.scatter(num_nodes_m6, tree_const_m6, c=['tab:orange']*len(num_nodes_m6))
plt.scatter(num_nodes_m7, tree_const_m7, c=['tab:brown']*len(num_nodes_m7))
plt.scatter(num_nodes_m8, tree_const_m8, c=['g']*len(num_nodes_m8))
#plt.scatter(num_nodes_m8b, tree_const_m8b, c=['y']*len(num_nodes_m8b))
plt.scatter(num_nodes_m9, tree_const_m9, c=['tab:red']*len(num_nodes_m9))
plt.scatter(num_nodes_m9b, tree_const_m9b, c=['tab:pink']*len(num_nodes_m9b))
plt.scatter(num_nodes_m10, tree_const_m10, c=['m']*len(num_nodes_m10))
plt.scatter(num_nodes_m10b, tree_const_m10b, c=['tab:purple']*len(num_nodes_m10b))
plt.title('ST Constant vs Number of Nodes for Real Data')
plt.xlabel('Number of Nodes')
plt.ylabel('ST Constant')
plt.xlim(0, 1000)
plt.legend(['real data', 'model 6', 'model 7', 'model 8', 'model 9', 'model 9b', 'model 10', 'model 10b'])
plt.savefig(f'imgs/st_cons/st_cons_zoomed_in_real_data_vs_models.png')
plt.show()


# Plot Real Data and Model 1
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# # First subplot: Original plot
# axes[0].scatter(num_nodes_real, tree_const_real, c=['b']*len(num_nodes_real))
# axes[0].scatter(num_nodes_m1, tree_const_m1, c=['r']*len(num_nodes_m1))
# axes[0].set_title('ST Constant vs Number of Nodes for Real Data')
# axes[0].set_xlabel('Number of Nodes')
# axes[0].set_ylabel('ST Constant')
# axes[0].legend(['real data', 'model 1'])

# # Second subplot: Same plot with xlim(0, 1000)
# axes[1].scatter(num_nodes_real, tree_const_real, c=['b']*len(num_nodes_real))
# axes[1].scatter(num_nodes_m1, tree_const_m1, c=['r']*len(num_nodes_m1))
# axes[1].set_title('ST Constant vs Number of Nodes (xlim=1000)')
# axes[1].set_xlabel('Number of Nodes')
# axes[1].set_ylabel('ST Constant')
# axes[1].set_xlim(0, 1000)
# axes[1].legend(['real data', 'model 1'])

# # Adjust layout and show plot
# plt.tight_layout()
# plt.savefig(f'imgs/st_cons/st_cons_real_data_vs_model1.png')
# plt.show()

# Plot Real Data and Model 2
plt.scatter(num_nodes_real, tree_const_real, c=['b']*len(num_nodes_real))
plt.scatter(num_nodes_m2, tree_const_m2, c=['g']*len(num_nodes_m2))
plt.title('ST Constant vs Number of Nodes')
plt.xlabel('Number of Nodes')
plt.ylabel('ST Constant')
plt.legend(['real data', 'model 2'])
plt.savefig(f'imgs/st_cons/st_cons_real_data_vs_model2.png')
plt.show()

# Plot Real Data and Model 3
plt.scatter(num_nodes_real, tree_const_real, c=['b']*len(num_nodes_real))
plt.scatter(num_nodes_m3, tree_const_m3, c=['y']*len(num_nodes_m3))
plt.title('ST Constant vs Number of Nodes')
plt.xlabel('Number of Nodes')
plt.ylabel('ST Constant')
plt.legend(['real data', 'model 3'])
plt.savefig(f'imgs/st_cons/st_cons_real_data_vs_model3.png')
plt.show()

# Plot Real Data and Model 4
plt.scatter(num_nodes_real, tree_const_real, c=['b']*len(num_nodes_real))
plt.scatter(num_nodes_m3, tree_const_m3, c=['y']*len(num_nodes_m3))
plt.scatter(num_nodes_m4, tree_const_m4, c=['m']*len(num_nodes_m4))
plt.scatter(num_nodes_m4b, tree_const_m4b, c=['tab:purple']*len(num_nodes_m4b))
plt.title('ST Constant vs Number of Nodes')
plt.xlabel('Number of Nodes')
plt.ylabel('ST Constant')
plt.legend(['real data', 'model 3', 'model 4', 'model 4b'])
plt.savefig(f'imgs/st_cons/st_cons_real_data_vs_model4.png')
plt.show()

# Plot Real Data and Model 5
plt.scatter(num_nodes_real, tree_const_real, c=['b']*len(num_nodes_real))
plt.scatter(num_nodes_m3, tree_const_m3, c=['y']*len(num_nodes_m3))
plt.scatter(num_nodes_m5, tree_const_m5, c=['r']*len(num_nodes_m5))
plt.scatter(num_nodes_m5b, tree_const_m5b, c=['tab:pink']*len(num_nodes_m5b))
plt.title('ST Constant vs Number of Nodes')
plt.xlabel('Number of Nodes')
plt.ylabel('ST Constant')
plt.xlim(0, 1000)
plt.legend(['real data', 'model 3', 'model 5', 'model 5b'])
plt.savefig(f'imgs/st_cons/st_cons_real_data_vs_model5.png')
plt.show()

# Plot Real Data and Model 6 and 7
plt.scatter(num_nodes_real, tree_const_real, c=['b']*len(num_nodes_real))
plt.scatter(num_nodes_m6, tree_const_m6, c=['tab:orange']*len(num_nodes_m6))
plt.scatter(num_nodes_m7, tree_const_m7, c=['tab:brown']*len(num_nodes_m7))
plt.title('ST Constant vs Number of Nodes for Real Data')
plt.xlabel('Number of Nodes')
plt.ylabel('ST Constant')
plt.xlim(0, 1000)
plt.legend(['real data', 'model 6', 'model 7'])
plt.savefig(f'imgs/st_cons/st_cons_real_data_vs_models6_7.png')
plt.show()