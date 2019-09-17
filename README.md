# ECON622 - Fall 2019
This is a graduate topics course in computational economics, with applications in datascience and machine learning.
# Course materials
- Get a [GitHub](www.github.com) ID and apply for the [Student Developer Pack](https://education.github.com/pack) to get further free features
- Consider clicking `Watch` at the top of this repository to see file changes
- (Optionally)  installing [GitHub Desktop](https://desktop.github.com) for easy downloads/updates of materials

## Accessing the VSE syzygy JupyterHub
1.  Login to https://vse.syzygy.ca/ with your CWL to ensure you can access our JupyterHub
2.  Click [Here](https://vse.syzygy.ca/jupyter/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FQuantEcon%2Fquantecon-notebooks-julia&urlpath=lab%2Ftree%2Fquantecon-notebooks-julia) to install the QuantEcon Julia Lectures there
    - Later you will need to do a local installation by following the [Getting Started](https://lectures.quantecon.org/jl/getting_started_julia/getting_started.html) but this is a better way to begin
    - For support with vse.syzygy.ca, email arnav.sood@ubc.ca
3. To automatically launch the QuantEcon lecture notes on vse.syzygy.ca
    - Open the lecture notes in a website (e.g. go to  [Introductory Examples](https://lectures.quantecon.org/jl/getting_started_julia/julia_by_example.html))
    - Hover your mouse over the button "jupyter notebook | run" at the top
    - When it pops up a configuration, choose `vse.syzygy.ca (UBC Only)` from the list, move your mouse to somewhere else on the screen
    - Now when you click on the "jupyter notebook | run" on any of the Julia lectures (no need to hover again), it will launch in our hub.
4. Download the extra notebooks from this repository with  [Here](https://vse.syzygy.ca/jupyter/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fubcecon%2FECON622_2019&urlpath=lab%2Ftree%2FECON622_2019%2F)
    - To update this repository when we create new notebooks, just click on that link again to clone.

In all cases, the reset a notebook, delete it and click on the launch of clone links again.

Most of the course will be taught using Julia, but we will briefly introduce Python (or R) for discussing topics where Julia is not ideal.

## Syllabus
See [Syllabus](syllabus.md) for more details


## Problem Sets
Problem sets should be submitted as a *single* Jupyter notebook on Canvas, with the code and output clean.
- **Problem Set 1** - Due Friday September 13th
  - Exercises 7, 8(a), and 8(b) in [Introductory Examples](https://lectures.quantecon.org/jl/getting_started_julia/julia_by_example.html)
  - Redo any exercise of your choice from 1 to 6 in [Introductory Examples](https://lectures.quantecon.org/jl/getting_started_julia/julia_by_example.html)
    - In all "Redo" look at the provided solutions afterwards, but still submit your code
  - Redo Exercise 5 in [Julia Essentials](https://lectures.quantecon.org/jl/getting_started_julia/julia_essentials.html) except
      - Pass in a range instead of `a`, `b`, and `n`, e.g. `nodes = -1.0:0.5:1.0`
      - Hint: try `step(nodes)`, `length(nodes)`, `nodes[1]` and `nodes[end]`
      - Also, instead of the `while` in the solution, find a better way to efficiently bracket the `x` in the nodes.  Hint: Type `?÷` into jupyter to explore quotients from Euclidean division
- **Problem Set 2** - Due Monday September 23th
  - Unlike the first assignment, we will start grading this on style and reproducibility rather than just on whether you get the correct numbers
  - Exercises 2, 3, and 4 in [Fundamental Types](https://lectures.quantecon.org/jl/getting_started_julia/fundamental_types.html)
  - Exercises 1, 2, and 3 in [Intro to Types](https://lectures.quantecon.org/jl/getting_started_julia/introduction_to_types.html)

<!-- 
- **Problem Set 3:**
    - Problems 4 to 6 in [Introduction to Types](https://lectures.quantecon.org/jl/getting_started_julia/introduction_to_types.html) 
    - Add something on Generic Programming as well?    
- ** Problem Set 4:** 
-  version_control.ipynb lecture, do exercises 1a, 1b, 1c, 2a, and 2b
In the (updated) testing.ipynb lecture, do exercise 1
    - For all of these, please prepare a ipynb notebook or something similar with links to the various PRs or screenshots with some evidence that you executed the steps.  No need to do much about the formatting
    - The easiest is certainly if you do all of this with public github repos, and then you can just provide links to the "evidence"
    - One more comment on this:  For the PRs, make sure to look at the style of the underlying code or text.  For example, if no punctuation is used anywhere in a document, then that is the style used.  Making style suggestions as PRs is not the best approach.
-->

## Lectures
1. **September 4th**: Environment and Introduction to Julia
    - [Julia Environment](https://lectures.quantecon.org/jl/getting_started_julia/julia_environment.html), [Introductory Examples](https://lectures.quantecon.org/jl/getting_started_julia/julia_by_example.html)

2. **September 9th**: Introduction and Variations on Fixed-points
   - [Introductory Examples](https://lectures.quantecon.org/jl/getting_started_julia/julia_by_example.html)
3. **September 11th**: Introduction to types
   -  Self-study: [Julia Essentials](https://lectures.quantecon.org/jl/getting_started_julia/julia_essentials.html)
   -  Self-study: [Fundamental Types](https://lectures.quantecon.org/jl/getting_started_julia/fundamental_types.html)
   -  Start [Intro to Generic Programming](https://lectures.quantecon.org/jl/getting_started_julia/introduction_to_types.html)
   -  Flip through the [Matlab - Julia Cheat Sheet](https://cheatsheets.quantecon.org/)
4. **September 16th**: Generic Programming
   -  Self-study: [Intro to Types](https://lectures.quantecon.org/jl/getting_started_julia/introduction_to_types.html)
   -  Self-study: [Generic Programming](https://lectures.quantecon.org/jl/more_julia/generic_programming.html)
   -  Self-study [General Packages](https://lectures.quantecon.org/jl/more_julia/general_packages.html)
   -  More on quadrature and numerical integration
5. **September 18th**:
6. **September 23th**:
7. **September 25th**:
8.  **September 30th**:
9.  **October 2nd**:
10. **October 7th**:
11. **October 9th**:
12. **October 14th**: NO CLASS (Thanksgiving)
13. **October 16th**: Clusters + "Enough Python to Read Glue Code"
14. **October 21th**: Webscraping and text
15. **October 23th**:
16. **October 28th**:
17. **October 30th**:
18. **November 4th**:
19. **November 6th**:
20. **November 11th**: NO CLASS (Remembrance Day)
21. **November 13th**:
22. **November 18th**:
23. **November 20th**:
24. **November 25th**:
25. **November 27th**: 
26. Final Project due mid December

