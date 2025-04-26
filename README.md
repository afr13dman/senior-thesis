# Random Graph Models for Dual Graphs

This repository includes comprehensive documentation and the scripts needed to reproduce all analyses and visualizations presented in the thesis Random Graphy Models for Dual Graphs by Anne Friedman, Scripps College Senior Thesis 2025. However, please note that the real graph data is not provided.

## Abstract

This project aims to better characterize dual graphs derived from state districting maps by developing random graph models that replicate their structural properties. Dual graphs provide a simplified way to represent districting maps, making it computationally feasible to analyze their structure. These representations enable researchers, legislators, and courts to assess district compactness, detect signs of gerrymandering, and generate alternative districting plans. A deeper understanding of the structural patterns of these dual graphs can help researchers choose or design more effective algorithms for redistricting analysis. The random graph models developed in this study serve as testbeds for evaluating algorithmic approaches to creating fair and unbiased districting plans. Through a variety of edge-adding and edge-removal strategies, the project identifies models that closely approximate two key statistics: average degree and the spanning tree constant.

## Explanation of Code
The computational workflow for this thesis is organized into the following files and folders:

1. `models.py` defines functions that reproduce all models created and analyzed in this thesis.
2. `spanning_trees.py` calculates the spanning tree constant and the logarithm of the spanning tree count for both the real data and all models. It also generates plots comparing the spanning tree counts of each model to the real data. For models with multiple variations, this file specifies the parameters used.
3. `model_deg_investigation.ipynb` investigates key structural properties such as planarity, connectivity, average degree, median degree, and maximum degree for Models Two through Twelve. For models with multiple variations, this notebook specifies the parameters used.
4. `visualize_models.ipynb` analyzes structural characteristics such as average degree and spanning tree constant, and visualizes the layouts of different random graph models.
5. `get_degs_st_cons.ipynb` is a streamlined version of `model_deg_investigation.ipynb` hat explores the structure of a single model, allowing for easier fine-tuning of parameters.
6. `model_one_base_vert_ipynb` investigates whether a relationship exists between the number of vertices and the probability base used in Model One.
7. `csv_to_latex_table.ipynb` converts CSV files into LaTeX tables for easy inclusion in the thesis document.
8. `real_graph_exploration` folder contains Python scripts, notebooks, and CSV files used to analyze the structure and properties of real-world dual graphs.
    - `avg_median_degree.py`: Calculates characteristics of real-world dual graphs derived from census tracts and block groups. 
    - `degree_exploration.ipynb`: Analyzes the degree distributions of dual graphs at the census tract and block group levels.
    - `connected_planar.ipynb`: Computes the percentage of graphs that are connected and planar for both census tracts and block groups.
9. `pkls` folder contains saved lists of spanning tree counts for real data and model-generated dual graphs.
10. `imgs` stores all saved images generated across the different Python files and notebooks.
11. `csv_files` stores CSV files containing summary statistics such as model averages and Model One-specific outputs.
12. `tex_files` folder contains LaTeX versions of selected tables, converted from CSV files, for easy integration into the thesis.