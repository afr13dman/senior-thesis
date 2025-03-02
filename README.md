# Senior Thesis in Data Science

_Week 1 Project Summary_

My thesis aims to generate random graphs that models dual graphs of maps used in redistricting. Dual graphs are graphs where each vertex corresponds to a region in the original map. For this project, the originals maps will be county, block, block group, and census tract maps for all states. Each vertex in the dual graph will share an edge with another vertex if those two vertices are adjacent in the original map. In order to be able to model dual graphs, I will investigate properties of these dual graphs to understand their characteristic to then model them in the random graphs and know what characteristics the the random graphs must contain. Since the county data is the smallest, has the least number of vertices, I will first use these when conducting my analysis and then I replicate my results with the other three types of maps: block, block group, and census tract. Finally, based on my analysis and results from all four types of maps, I will generate my conclusions.


## Daily Notes

_02-14-2025_

Wrote to code to generate a random graph with edges created based on a probability $n^dist$ where dist is the distance between vertice u and v. I created four graphs with $n = 2,5,10,15$. Below are the resulting graphs in that order.

<img src=imgs/prob_two.png/>

<img src=imgs/prob_five.png/>

<img src=imgs/prob_ten.png/>

<img src=imgs/prob_fifteen.png/>

Based on this, the probability definetly has to be greater than 5 because the first two graphs had too many edges.

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

_03-01-2025_

I found the average and median degree for each state at both tract and block group level. Then for each level, I found the average across states. The average median degrees for tract and block groups was 5.0, while the average average degree was 5.398 at tract level and 5.441 at block group level. I think I should use average degree at around 5.4 and redo model two.

