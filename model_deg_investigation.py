'''
This file explores the planarity, connectivity, average degree, median degree, and max degree 
for Models 2 through 12 for graphs with a specified number of vertices and random seed.
'''

import importlib
import models
importlib.reload(models)

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from statistics import median

def max_degree(G):
    # Initialize max_degree
    max_degree = -1

    # Iterate over all nodes and their degrees
    for degree in G.degree():
        if degree[1] > max_degree:
            max_degree = degree[1]
    
    return max_degree

# Create dataset to save values
model_avgs = pd.DataFrame()

# Set up dataset of random seeds and number of vertices to explore
df = pd.read_csv("model_one_desired_avg_deg.csv")
df = df[["num_vertices", "rand_seed"]].drop_duplicates()
new_vert = [200, 400, 600, 800, 1000, 1200, 1400, 1600]
for vert in new_vert:
    for seed in df["rand_seed"].unique()[0:3]:
        new_row = pd.DataFrame([{'num_vertices': vert, 'rand_seed': seed}])
        df = pd.concat([df, new_row], ignore_index=True)

#############
#  model 2  #
#############

planar = []
connected = []
avg_deg = []
median_deg = []
max_deg = []

for index, row in df.iterrows():
    n = int(row["num_vertices"])
    rs = int(row["rand_seed"])
    G = models.model_two(n, rs)

    # Planar?
    planar.append(nx.check_planarity(G)[0])
    
    # Connected?
    connected.append(nx.is_connected(G))

    # Calculate avg degree of graph
    avg_deg.append(2 * G.number_of_edges() / G.number_of_nodes())
    
    # Calculate median degree of graph
    degrees = sorted([degree for _, degree in G.degree()], reverse=False)
    median_deg.append(median(degrees))

    # Calculate max degree
    max_deg.append(max_degree(G))

model_two_df = df.assign(planar=planar, connected=connected, 
                               avg_deg=avg_deg, median_deg=median_deg, max_deg=max_deg)

# Calculate the average of planar, connected, median_deg, max_deg
columns_avg = model_two_df[['planar', 'connected', 'avg_deg', 'median_deg', 'max_deg']].mean()
model_avgs = pd.concat([model_avgs, columns_avg.rename('Model 3')], axis=1)

#############
#  model 3  #
#############

planar = []
connected = []
avg_deg = []
median_deg = []
max_deg = []

for index, row in df.iterrows():
    n = int(row["num_vertices"])
    rs = int(row["rand_seed"])
    G = models.model_three(n, rs)

    # Planar?
    planar.append(nx.check_planarity(G)[0])
    
    # Connected?
    connected.append(nx.is_connected(G))

    # Calculate avg degree of graph
    avg_deg.append(2 * G.number_of_edges() / G.number_of_nodes())
    
    # Calculate median degree of graph
    degrees = sorted([degree for _, degree in G.degree()], reverse=False)
    median_deg.append(median(degrees))

    # Calculate max degree
    max_deg.append(max_degree(G))

model_three_df = df.assign(planar=planar, connected=connected, 
                               avg_deg=avg_deg, median_deg=median_deg, max_deg=max_deg)

# Calculate the average of planar, connected, median_deg, max_deg
columns_avg = model_three_df[['planar', 'connected', 'avg_deg', 'median_deg', 'max_deg']].mean()
model_avgs = pd.concat([model_avgs, columns_avg.rename('Model 3')], axis=1)

#############
#  model 4  #
#############

planar = []
connected = []
avg_deg = []
median_deg = []
max_deg = []

for index, row in df.iterrows():
    n = int(row["num_vertices"])
    rs = int(row["rand_seed"])
    G = models.model_dt_with_removal(n, rs)

    # Planar?
    planar.append(nx.check_planarity(G)[0])
    
    # Connected?
    connected.append(nx.is_connected(G))

    # Calculate avg degree of graph
    avg_deg.append(2 * G.number_of_edges() / G.number_of_nodes())
    
    # Calculate median degree of graph
    degrees = sorted([degree for _, degree in G.degree()], reverse=False)
    median_deg.append(median(degrees))

    # Calculate max degree
    max_deg.append(max_degree(G))

model_four_df = df.assign(planar=planar, connected=connected, 
                               avg_deg=avg_deg, median_deg=median_deg, max_deg=max_deg)

# Calculate the average of planar, connected, median_deg, max_deg
columns_avg = model_four_df[['planar', 'connected', 'avg_deg', 'median_deg', 'max_deg']].mean()
model_avgs = pd.concat([model_avgs, columns_avg.rename('Model 4')], axis=1)

##############
#  model 4b  #
##############

planar = []
connected = []
avg_deg = []
median_deg = []
max_deg = []

for index, row in df.iterrows():
    n = int(row["num_vertices"])
    rs = int(row["rand_seed"])
    G = models.model_dt_with_removal(n, rs, 0.4)

    # Planar?
    planar.append(nx.check_planarity(G)[0])
    
    # Connected?
    connected.append(nx.is_connected(G))

    # Calculate avg degree of graph
    avg_deg.append(2 * G.number_of_edges() / G.number_of_nodes())
    
    # Calculate median degree of graph
    degrees = sorted([degree for _, degree in G.degree()], reverse=False)
    median_deg.append(median(degrees))

    # Calculate max degree
    max_deg.append(max_degree(G))

model_fourb_df = df.assign(planar=planar, connected=connected, 
                               avg_deg=avg_deg, median_deg=median_deg, max_deg=max_deg)

# Calculate the average of planar, connected, median_deg, max_deg
columns_avg = model_fourb_df[['planar', 'connected', 'avg_deg', 'median_deg', 'max_deg']].mean()
model_avgs = pd.concat([model_avgs, columns_avg.rename('Model 4b')], axis=1)

#############
#  model 5  #
#############

planar = []
connected = []
avg_deg = []
median_deg = []
max_deg = []

for index, row in df.iterrows():
    n = int(row["num_vertices"])
    rs = int(row["rand_seed"])
    G = models.model_dt_with_removal_and_add(n, rs)

    # Planar?
    planar.append(nx.check_planarity(G)[0])
    
    # Connected?
    connected.append(nx.is_connected(G))

    # Calculate avg degree of graph
    avg_deg.append(2 * G.number_of_edges() / G.number_of_nodes())
    
    # Calculate median degree of graph
    degrees = sorted([degree for _, degree in G.degree()], reverse=False)
    median_deg.append(median(degrees))

    # Calculate max degree
    max_deg.append(max_degree(G))

model_five_df = df.assign(planar=planar, connected=connected, 
                               avg_deg=avg_deg, median_deg=median_deg, max_deg=max_deg)

# Calculate the average of planar, connected, median_deg, max_deg
columns_avg = model_five_df[['planar', 'connected', 'avg_deg', 'median_deg', 'max_deg']].mean()
model_avgs = pd.concat([model_avgs, columns_avg.rename('Model 5')], axis=1)

##############
#  model 5b  #
##############

planar = []
connected = []
avg_deg = []
median_deg = []
max_deg = []

for index, row in df.iterrows():
    n = int(row["num_vertices"])
    rs = int(row["rand_seed"])
    G = models.model_dt_with_removal_and_add(n, rs, remove_prob=0.4)

    # Planar?
    planar.append(nx.check_planarity(G)[0])
    
    # Connected?
    connected.append(nx.is_connected(G))

    # Calculate avg degree of graph
    avg_deg.append(2 * G.number_of_edges() / G.number_of_nodes())
    
    # Calculate median degree of graph
    degrees = sorted([degree for _, degree in G.degree()], reverse=False)
    median_deg.append(median(degrees))

    # Calculate max degree
    max_deg.append(max_degree(G))

model_fiveb_df = df.assign(planar=planar, connected=connected, 
                               avg_deg=avg_deg, median_deg=median_deg, max_deg=max_deg)

# Calculate the average of planar, connected, median_deg, max_deg
columns_avg = model_fiveb_df[['planar', 'connected', 'avg_deg', 'median_deg', 'max_deg']].mean()
model_avgs = pd.concat([model_avgs, columns_avg.rename('Model 5b')], axis=1)

#############
#  model 6  #
#############

planar = []
connected = []
avg_deg = []
median_deg = []
max_deg = []

for index, row in df.iterrows():
    n = int(row["num_vertices"])
    rs = int(row["rand_seed"])
    G = models.model_dt_add_short_edges(n, rs)

    # Planar?
    planar.append(nx.check_planarity(G)[0])
    
    # Connected?
    connected.append(nx.is_connected(G))

    # Calculate avg degree of graph
    avg_deg.append(2 * G.number_of_edges() / G.number_of_nodes())
    
    # Calculate median degree of graph
    degrees = sorted([degree for _, degree in G.degree()], reverse=False)
    median_deg.append(median(degrees))

    # Calculate max degree
    max_deg.append(max_degree(G))

model_six_df = df.assign(planar=planar, connected=connected, 
                               avg_deg=avg_deg, median_deg=median_deg, max_deg=max_deg)

# Calculate the average of planar, connected, median_deg, max_deg
columns_avg = model_six_df[['planar', 'connected', 'avg_deg', 'median_deg', 'max_deg']].mean()
model_avgs = pd.concat([model_avgs, columns_avg.rename('Model 6')], axis=1)

#############
#  model 7  #
#############

planar = []
connected = []
avg_deg = []
median_deg = []
max_deg = []

for index, row in df.iterrows():
    n = int(row["num_vertices"])
    rs = int(row["rand_seed"])
    G = models.model_dt_add_short_remove_long(n, rs)

    # Planar?
    planar.append(nx.check_planarity(G)[0])
    
    # Connected?
    connected.append(nx.is_connected(G))

    # Calculate avg degree of graph
    avg_deg.append(2 * G.number_of_edges() / G.number_of_nodes())
    
    # Calculate median degree of graph
    degrees = sorted([degree for _, degree in G.degree()], reverse=False)
    median_deg.append(median(degrees))

    # Calculate max degree
    max_deg.append(max_degree(G))

model_seven_df = df.assign(planar=planar, connected=connected, 
                               avg_deg=avg_deg, median_deg=median_deg, max_deg=max_deg)

# Calculate the average of planar, connected, median_deg, max_deg
columns_avg = model_seven_df[['planar', 'connected', 'avg_deg', 'median_deg', 'max_deg']].mean()
model_avgs = pd.concat([model_avgs, columns_avg.rename('Model 7')], axis=1)

#############
#  model 8  #
#############

planar = []
connected = []
avg_deg = []
median_deg = []
max_deg = []

for index, row in df.iterrows():
    n = int(row["num_vertices"])
    rs = int(row["rand_seed"])
    G = models.model_eight(n, rs)

    # Planar?
    planar.append(nx.check_planarity(G)[0])
    
    # Connected?
    connected.append(nx.is_connected(G))

    # Calculate avg degree of graph
    avg_deg.append(2 * G.number_of_edges() / G.number_of_nodes())
    
    # Calculate median degree of graph
    degrees = sorted([degree for _, degree in G.degree()], reverse=False)
    median_deg.append(median(degrees))

    # Calculate max degree
    max_deg.append(max_degree(G))

model_eight_df = df.assign(planar=planar, connected=connected, 
                             avg_deg=avg_deg, median_deg=median_deg, max_deg=max_deg)

# Calculate the average of planar, connected, median_deg, max_deg
columns_avg = model_eight_df[['planar', 'connected', 'avg_deg', 'median_deg', 'max_deg']].mean()
model_avgs = pd.concat([model_avgs, columns_avg.rename('Model 8')], axis=1)

#############
#  model 9  #
#############

planar = []
connected = []
avg_deg = []
median_deg = []
max_deg = []

for index, row in df.iterrows():
    n = int(row["num_vertices"])
    rs = int(row["rand_seed"])
    G = models.model_dt_with_removal_add_shortest_edges(n, rs)

    # Planar?
    planar.append(nx.check_planarity(G)[0])
    
    # Connected?
    connected.append(nx.is_connected(G))

    # Calculate avg degree of graph
    avg_deg.append(2 * G.number_of_edges() / G.number_of_nodes())
    
    # Calculate median degree of graph
    degrees = sorted([degree for _, degree in G.degree()], reverse=False)
    median_deg.append(median(degrees))

    # Calculate max degree
    max_deg.append(max_degree(G))

model_nine_df = df.assign(planar=planar, connected=connected, 
                             avg_deg=avg_deg, median_deg=median_deg, max_deg=max_deg)

# Calculate the average of planar, connected, median_deg, max_deg
columns_avg = model_nine_df[['planar', 'connected', 'avg_deg', 'median_deg', 'max_deg']].mean()
model_avgs = pd.concat([model_avgs, columns_avg.rename('Model 9')], axis=1)

##############
#  model 9b  #
##############

planar = []
connected = []
avg_deg = []
median_deg = []
max_deg = []

for index, row in df.iterrows():
    n = int(row["num_vertices"])
    rs = int(row["rand_seed"])
    G = models.model_dt_with_removal_add_shortest_edges(n, rs, remove_prob=0.4)

    # Planar?
    planar.append(nx.check_planarity(G)[0])
    
    # Connected?
    connected.append(nx.is_connected(G))

    # Calculate avg degree of graph
    avg_deg.append(2 * G.number_of_edges() / G.number_of_nodes())
    
    # Calculate median degree of graph
    degrees = sorted([degree for _, degree in G.degree()], reverse=False)
    median_deg.append(median(degrees))

    # Calculate max degree
    max_deg.append(max_degree(G))

model_nineb_df = df.assign(planar=planar, connected=connected, 
                             avg_deg=avg_deg, median_deg=median_deg, max_deg=max_deg)

# Calculate the average of planar, connected, median_deg, max_deg
columns_avg = model_nineb_df[['planar', 'connected', 'avg_deg', 'median_deg', 'max_deg']].mean()
model_avgs = pd.concat([model_avgs, columns_avg.rename('Model 9b')], axis=1)

##############
#  model 10  #
##############

planar = []
connected = []
avg_deg = []
median_deg = []
max_deg = []

for index, row in df.iterrows():
    n = int(row["num_vertices"])
    rs = int(row["rand_seed"])
    G = models.model_dt_add_shortest_edges_remove_rand(n, rs)

    # Planar?
    planar.append(nx.check_planarity(G)[0])
    
    # Connected?
    connected.append(nx.is_connected(G))

    # Calculate avg degree of graph
    avg_deg.append(2 * G.number_of_edges() / G.number_of_nodes())
    
    # Calculate median degree of graph
    degrees = sorted([degree for _, degree in G.degree()], reverse=False)
    median_deg.append(median(degrees))

    # Calculate max degree
    max_deg.append(max_degree(G))

model_ten_df = df.assign(planar=planar, connected=connected, 
                             avg_deg=avg_deg, median_deg=median_deg, max_deg=max_deg)

# Calculate the average of planar, connected, median_deg, max_deg
columns_avg = model_ten_df[['planar', 'connected', 'avg_deg', 'median_deg', 'max_deg']].mean()
model_avgs = pd.concat([model_avgs, columns_avg.rename('Model 10')], axis=1)

###############
#  model 10b  #
###############

planar = []
connected = []
avg_deg = []
median_deg = []
max_deg = []

for index, row in df.iterrows():
    n = int(row["num_vertices"])
    rs = int(row["rand_seed"])
    G = models.model_dt_add_shortest_edges_remove_rand(n, rs, remove_prob=0.4)

    # Planar?
    planar.append(nx.check_planarity(G)[0])
    
    # Connected?
    connected.append(nx.is_connected(G))

    # Calculate avg degree of graph
    avg_deg.append(2 * G.number_of_edges() / G.number_of_nodes())
    
    # Calculate median degree of graph
    degrees = sorted([degree for _, degree in G.degree()], reverse=False)
    median_deg.append(median(degrees))

    # Calculate max degree
    max_deg.append(max_degree(G))

model_tenb_df = df.assign(planar=planar, connected=connected, 
                             avg_deg=avg_deg, median_deg=median_deg, max_deg=max_deg)

# Calculate the average of planar, connected, median_deg, max_deg
columns_avg = model_tenb_df[['planar', 'connected', 'avg_deg', 'median_deg', 'max_deg']].mean()
model_avgs = pd.concat([model_avgs, columns_avg.rename('Model 10b')], axis=1)

##############
#  model 11  #
##############

planar = []
connected = []
avg_deg = []
median_deg = []
max_deg = []

for index, row in df.iterrows():
    n = int(row["num_vertices"])
    rs = int(row["rand_seed"])
    G = models.model_two_rand_removal(n, rs)

    # Planar?
    planar.append(nx.check_planarity(G)[0])
    
    # Connected?
    connected.append(nx.is_connected(G))

    # Calculate avg degree of graph
    avg_deg.append(2 * G.number_of_edges() / G.number_of_nodes())
    
    # Calculate median degree of graph
    degrees = sorted([degree for _, degree in G.degree()], reverse=False)
    median_deg.append(median(degrees))

    # Calculate max degree
    max_deg.append(max_degree(G))

model_eleven_df = df.assign(planar=planar, connected=connected, 
                             avg_deg=avg_deg, median_deg=median_deg, max_deg=max_deg)

# Calculate the average of planar, connected, median_deg, max_deg
columns_avg = model_eleven_df[['planar', 'connected', 'avg_deg', 'median_deg', 'max_deg']].mean()
model_avgs = pd.concat([model_avgs, columns_avg.rename('Model 11')], axis=1)

###############
#  model 11b  #
###############

planar = []
connected = []
avg_deg = []
median_deg = []
max_deg = []

for index, row in df.iterrows():
    n = int(row["num_vertices"])
    rs = int(row["rand_seed"])
    G = models.model_two_rand_removal(n, rs, scaling_factor=9, remove_prob=0.4)

    # Planar?
    planar.append(nx.check_planarity(G)[0])
    
    # Connected?
    connected.append(nx.is_connected(G))

    # Calculate avg degree of graph
    avg_deg.append(2 * G.number_of_edges() / G.number_of_nodes())
    
    # Calculate median degree of graph
    degrees = sorted([degree for _, degree in G.degree()], reverse=False)
    median_deg.append(median(degrees))

    # Calculate max degree
    max_deg.append(max_degree(G))

model_elevenb_df = df.assign(planar=planar, connected=connected, 
                             avg_deg=avg_deg, median_deg=median_deg, max_deg=max_deg)

# Calculate the average of planar, connected, median_deg, max_deg
columns_avg = model_elevenb_df[['planar', 'connected', 'avg_deg', 'median_deg', 'max_deg']].mean()
model_avgs = pd.concat([model_avgs, columns_avg.rename('Model 11b')], axis=1)

###############
#  model 11c  #
###############

planar = []
connected = []
avg_deg = []
median_deg = []
max_deg = []

for index, row in df.iterrows():
    n = int(row["num_vertices"])
    rs = int(row["rand_seed"])
    G = models.model_two_rand_removal(n, rs, scaling_factor=14, remove_prob=0.6)

    # Planar?
    planar.append(nx.check_planarity(G)[0])
    
    # Connected?
    connected.append(nx.is_connected(G))

    # Calculate avg degree of graph
    avg_deg.append(2 * G.number_of_edges() / G.number_of_nodes())
    
    # Calculate median degree of graph
    degrees = sorted([degree for _, degree in G.degree()], reverse=False)
    median_deg.append(median(degrees))

    # Calculate max degree
    max_deg.append(max_degree(G))

model_elevenc_df = df.assign(planar=planar, connected=connected, 
                             avg_deg=avg_deg, median_deg=median_deg, max_deg=max_deg)

# Calculate the average of planar, connected, median_deg, max_deg
columns_avg = model_elevenc_df[['planar', 'connected', 'avg_deg', 'median_deg', 'max_deg']].mean()
model_avgs = pd.concat([model_avgs, columns_avg.rename('Model 11c')], axis=1)

##############
#  model 12  #
##############

planar = []
connected = []
avg_deg = []
median_deg = []
max_deg = []

for index, row in df.iterrows():
    n = int(row["num_vertices"])
    rs = int(row["rand_seed"])
    G = models.model_eight_order_switched(n, rs)

    # Planar?
    planar.append(nx.check_planarity(G)[0])
    
    # Connected?
    connected.append(nx.is_connected(G))

    # Calculate avg degree of graph
    avg_deg.append(2 * G.number_of_edges() / G.number_of_nodes())
    
    # Calculate median degree of graph
    degrees = sorted([degree for _, degree in G.degree()], reverse=False)
    median_deg.append(median(degrees))

    # Calculate max degree
    max_deg.append(max_degree(G))

model_twelve_df = df.assign(planar=planar, connected=connected, 
                             avg_deg=avg_deg, median_deg=median_deg, max_deg=max_deg)

# Calculate the average of planar, connected, median_deg, max_deg
columns_avg = model_twelve_df[['planar', 'connected', 'avg_deg', 'median_deg', 'max_deg']].mean()
model_avgs = pd.concat([model_avgs, columns_avg.rename('Model 12')], axis=1)

#################
#  SAVE to CSV  #
#################

model_avgs = model_avgs.round(2)
model_avgs = model_avgs.T
model_avgs

model_avgs.to_csv("csv_files/model_avgs.csv")