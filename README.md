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