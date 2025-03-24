# Senior Thesis in Data Science

_Week 1 Project Summary_

My thesis aims to generate random graphs that models dual graphs of maps used in redistricting. Dual graphs are graphs where each vertex corresponds to a region in the original map. For this project, the originals maps will be county, block, block group, and census tract maps for all states. Each vertex in the dual graph will share an edge with another vertex if those two vertices are adjacent in the original map. In order to be able to model dual graphs, I will investigate properties of these dual graphs to understand their characteristic to then model them in the random graphs and know what characteristics the the random graphs must contain. Since the county data is the smallest, has the least number of vertices, I will first use these when conducting my analysis and then I replicate my results with the other three types of maps: block, block group, and census tract. Finally, based on my analysis and results from all four types of maps, I will generate my conclusions.


## Daily Notes

_02-14-2025_

Wrote to code to generate a random graph with edges created based on a probability $n^{dist}$ where dist is the distance between vertice u and v. I created four graphs with $n = 2,5,10,15$. Based on this, the probability definetly has to be greater than 5 because the first two graphs had too many edges.

Next step:
- Read three books found
- Generate 1-3 ideas of other ways to model random graphs
- Create a list of properties/characteristics of random graphs as well as dual graphs

_02-21-2025_

I looked at three different models:
- Model One: Add an edge based on a probability
- Model Two: Add the (5.2n) shortest edges where n is the number of vertices
- Model Three: Delaunay Triangulation

For each model, I explored the resulting graphs for 20, 30, 40, and 50 vertices with the random seed set to 12, 47, and 50. With Model One, I found the base which would create a graph that had an average degree of around 5.2. I found that some of the graphs created were not connected, but these occured with when the number of vertices was higher. Meanwhile, for Model Two and Model Three, all graphs were connected. I had initially thought that Model Two may create unconnected graphs, which I did not find, but potentially there is a seed which will create an unconnected graph.

For Model One, I found that the base had to be larger than the number of vertices, and as the number of vertices increases, the base had to increase exponentially, but it also depended greatly on the random seed.

I did not calculate the average degree for Models Two and Three.

I wonder if there is a way to a model that combines two or three of these models. 

Also, articles about Delaunay Triangulation:
- https://ianthehenry.com/posts/delaunay/
- https://gwlucastrig.github.io/TinfourDocs/DelaunayIntro/index.html

_03-01-2025_

I found the average and median degree for each state at both tract and block group level. Then for each level, I found the average across states. The average median degrees for tract and block groups was 5.0, while the average average degree was 5.398 at tract level and 5.441 at block group level. I think I should use average degree at around 5.4 and redo model two.

Things I did:
- Created `models.py` in coding_models folder, which includes four functions: one to create a random graph based on a given number of vertices and random seed. Then three others, each which creates a graph based on model 1, model 2, and model 3, respectively.
- Created a python notebook, `model_investigation.ipynb`, which uses the functions from `models.py` to create graphs based and then check if planar and connected and then plot them. This speeds up my plotting graphs and organizing them into the powerpoint that I was going last week.
- Wrote `avg_median_degree.py` which computes the average degree and median degree for the real data at tract and block group. Then `degree_exploration.ipynb` find the average of each. This will help when creating the model graphs to know what avg degree I want to aim for.
- Used Prof Cannon research students code plus edited it to plot spanning trees of real data and my models in `spanning_trees.py`.
    - not working right now...

Properties of Dual Graphs / Other Ways to assess our models:
- Degree Distribution: Compare the degree distribution of your random graphs to that of the real dual graphs. Many geographic dual graphs tend to have a heavy-tailed or nearly uniform degree distribution.
    - Avg, max, median
    - Histogram of the degrees
- Cluster Coefficient: Measures the likelihood that two adjacent nodes of a graph are also connected. Real-world geographic dual graphs often exhibit higher clustering due to adjacency relationships.
- Graph Assortativity: Measures whether high-degree nodes tend to be connected to other high-degree nodes. Many geographic networks exhibit neutral or negative assortativity.

_03-06-2025_

- Updated model 2 to add the 5.4n shortest edges
- Explored more of model 1, looking at different random seeds and the base needed to get an avg degree of 5.4 and more.
    - Need to see if I can find a relationship between them...
- Plotting spanning trees: Model 1 has the closest ST constant
- Planar and Connected: Only Model 3 is both. Model 1 and 2 are not planar. Model 2 has always been connected except for once with 40 vertices and random seed 23. Model 1 is connected about two thirds of the time.
- Added Max Degree calculation: Model 2 has the closest avg max degree but still not close real data.


_03-23-2025_

- Updated model 2 to add (5.4/2)n shortest edges
- Spanning trees:
<img src=imgs/st_cons/st_cons_real_data_vs_models.png>

Now I zoomed in to better see how the spanning tree constant for the models changed.
<img src=imgs/st_cons/st_cons_zoomed_in_real_data_vs_models.png>

- SP notes:
    - Model 1 and 2 have the closest st constant to real data. Model 3 is the furthest away but removing edges randomly brings the st constant closer to read data.

- For Model 1 and Model 2, if the graph was not connected, I chose the largest connected subgraph so that the graphs would be connected. Now all models have connected graphs.

- Tried to run more trials with larger number of vertices for Model 2 and 3. With number of vertices [200, 400, 600, 800], it was taking a ton of time to run so I had to pause and stop and will come back to it later.
    - It takes a while for Model 2, not model 3 or model 4. I think it's because it has to order the edges by distance
        - 7 seconds with no adds
        - 20 seconds to add three 200 num vertices graph
        - 3 minutes to add three 200 num vertices and three 400 num vertices
        - 15 minutes to add three 200 num vertices, three 400 num vertices, and three 600 # vertices
        - About an hour to add three 200 num vertices, three 400 num vertices, three 600 # vertices, and three 800 # vertices

- Added model 4: delaunay triangulation with randomly removing edges, basically model 3 with edge removal. It is always planar and connected
    - With remove probability of 0.2: Too small average degree and median degree (close to 3.something). Max degree is also small (around 6). ST cons way too high
    - With remove probability of 0.4: ST cons is closer to real world data, but avg deg, med deg, and max deg much smaller.

- Add Model 1 plots, maybe exponential but the log(base) vs. # of vertices doesn't exactly have a linear relationship, but it's the closests out of all the possibilities. Polynomial is the next closests.
