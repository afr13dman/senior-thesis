# Tasks

## Week 1
- [x] Read through paper
- [x] Look at GitHub
- [x] Set up a workflow— set up my own GitHub
    - [x] start with county graphs
    - [x] loop through all graphs and see if i can get number of vertices
    - [x] how will I store graphs and data
    - [ ] ~~try to make missing graphs~~
        - [ ] ~~find shape files and turn into graphs and then write to json file~~
    - [ ] ~~start generating some ideas~~
        - [ ] ~~we want to build random graphs that somehow resemble dual graphs~~
    - [x] write a paragraph summary of what my project is

## Week 2/3
- [ ] try to make missing graphs
    - [ ] find shape files and turn into graphs and then write to json file
- [ ] ~~properties of dual graphs~~
- [ ] start generating some ideas: we want to build random graphs that somehow resemble dual graphs
    - [ ] research ideas: google scholar, textbooks at the library (random graphs)
        - what are some random graph models? what is being done? what are ways to characterize graphs?
    - [ ] bring 2-4 ideas for modeling random graphs and 2-4 ways to assess them
- [x] look at the model prof cannon suggested
    - [x] networkx to store vertices and edges
    - [x] set a random seed
- [x] write up motivation for current thesis (ish, thought and talked about it)

## Week 4
- [x] Create a thesis block time during the week (10 to 12 hours) (ish)
- [x] Create a new step list after each day/work time

## Week 5
- [x] Investigate model one more
- [x] Start model two investigation
- [x] Start delaunay triangulation model
- [ ] ~~Read some of the black book~~

## Week 6
- [x] Plot Model 1 without lat and long coordinates
- [x] Plot Model 2 without lat and long coordinates
- [x] Find actual avg/median degree of all dual graphs
- [x] Check if graphs are planar and connected
- [x] More trials for M1 to see if I can find a relationship between base and number of vectices
- [x] Spanning trees as another way to assess our models
    - [x] and what are other properties of dual graphs and ways to assess our models
- [ ] Read some of the black book

## Week 7

## Week 8
- [x] Model 2 to add the (5.4/2)n shortest edges
- [x] Spanning trees zoom in to 0 to 1000 on x-axis
- [x] Model 1: remove the vertices that aren't connected to the main chunk to make it connected
- [x] For Model 1, plot base against # of vertices
    - [x] log(base) vs. # vertices: if original curve is exponential, this will be linear
    - [x] log(base) vs. log(# vertices): if original curve is polynomial, this will be linear
    - [x] base vs. log(# vertices): if original curve is logarithmic, this will be linear
    - (log base 2)
- [x] Running more trials with larger number of vertices (Model 2 and 3)
    - [x] About an hour to add three 200 num vertices, three 400 num vertices, three 600 # vertices, and three 800 # vertices for Model 2
- [x] Spend 3 hours on how to merge parts of models together to get the properties we want
    - [x] Take delaunay triangulation and randomly remove some edges

## Week 9
- [ ] time.time to get clock time for model 2 to see what is running long
- [x] Delaunay triangulation add more shortest edges, but will probably be too many spanning trees
- [ ] DT but remove longest edges or only smallest edges
- [x] DT add shortest edges and remove longest edges
- [x] DT remove random edges and add random edges
- [ ] Spend 1-3 hours writing: EXPLAIN DUAL GRAPHS AND MODELS
    - Write, don't try to be perfect
    - Break it down into small chunks: write for 15 minutes, take a break, and then see what is achievable
    - Take frequent breaks

# Random notes
Our graphs:
- not many high degree vertices (maybe max 50)
- planar or close to planar




