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
    - For support with vse.syzygy.ca, email me@arnavsood.com
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
  - Exercise 7 in [Julia Essentials](https://lectures.quantecon.org/jl/getting_started_julia/julia_essentials.html)
- **Problem Set 2** - Due Monday September 23th
  - Unlike the first assignment, we will start grading this on style and reproducibility rather than just on whether you get the correct numbers.  See the [Digression on Style and Naming](https://lectures.quantecon.org/jl/getting_started_julia/introduction_to_types.html#A-Digression-on-Style-and-Naming)
  - Exercises 2, 3, and 4 in [Fundamental Types](https://lectures.quantecon.org/jl/getting_started_julia/fundamental_types.html)
  - Exercises 1, 2, and 3 in [Introduction to Types](https://lectures.quantecon.org/jl/getting_started_julia/introduction_to_types.html)
- **Problem Set 3** - Due Saturday September 28th
  - Exercises 4, 5, 6 in [Introduction to Types](https://lectures.quantecon.org/jl/getting_started_julia/introduction_to_types.html)
- **Problem Set 4:** Due Saturday, October 5th
  - Exercise 1 in [Generic Programming](https://lectures.quantecon.org/jl/more_julia/generic_programming.html)
  - (Optional) Exercise 2 in [Generic Programming](https://lectures.quantecon.org/jl/more_julia/generic_programming.html)
  - Exercises 1a, 1b, 1c, 2a, and 2b in [Git and Github](https://lectures.quantecon.org/jl/more_julia/version_control.html)
    - For the git/github in your ipynb notebook add links to the various PRs or screenshots with some evidence that you executed the steps.  No need to do much about the formatting
     - The easiest is certainly if you do all of this with public github repos, and then you can just provide links to the "evidence"
    - One more comment on this:  For the PRs, make sure to look at the style of the underlying code or text.  For example, if no punctuation is used anywhere in a document, then that is the style used.  Making style suggestions as PRs is not the best approach.
- **Problem Set 5:** Due Saturday, October 12th
  - Exercise 1 in [Testing and Packages](https://julia.quantecon.org/more_julia/testing.html)
  - Exercise 1, 2a, and 2b in [Numerical Linear Algebra](https://github.com/ubcecon/ECON622_2019/blob/master/notebooks/numerical_linear_algebra.ipynb)
  
- **Problem Set 6:** Due Monday, October 28th
  - Complete one of the exercises from [optimization algorithms](https://schrimpf.github.io/AnimatedOptimization.jl/exercises/) Turn in a link to a public git repo containing your work (preferred) or a jupyter notebook.
  
- **Problem Set 7:** Due Monday, November 4th
  - Work on one of the issues in [GMMInference.jl](https://github.com/schrimpf/GMMInference.jl/issues). 
    - If you have a GMM model you're interested in, Issue #7 would be a good choice
    - If you are interested in econometric theory, issues #5 and the second bullet of #8 are good and will require some research
    - If you like thinking about code organization and package design, #2 or #6 are relevant
  - As with the previous assignment, you need not complete the task; make whatever progress you can in 6 hours or so. If you want your work to be added to the repository, either make a pull request or say so on whatever you turn in. 

- **Problem Set 8:** Due Wednesday, November 13th
  - Improve the performance of a piece of code. Take some code from a package, previous assignment, or lecture and attempt to improve its performance. Include benchmarks of the initial version and your modified version. Briefly describe the things you tried. 

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
4. **September 16th**
   -  Self-study: [Intro to Generic Programming](https://lectures.quantecon.org/jl/getting_started_julia/introduction_to_types.html)
   -  Self-study: [Generic Programming](https://lectures.quantecon.org/jl/more_julia/generic_programming.html)
5. **September 18th**
   -  [Generic Programming](https://lectures.quantecon.org/jl/more_julia/generic_programming.html)
   -  Self-study: [General Packages](https://lectures.quantecon.org/jl/more_julia/general_packages.html)
   -  Self-study: [Data and Statistical Packages](https://lectures.quantecon.org/jl/more_julia/data_statistical_packages.html)
   -  Notes on [Quadrature](https://nbviewer.jupyter.org/github/ubcecon/ECON622_2019/blob/master/notebooks/quadrature.ipynb) applying generic programming
6. **September 23th**
   -  Self-study: [Linear Algebra](https://lectures.quantecon.org/jl/tools_and_techniques/linear_algebra.html)
   -  Self-study: [Orthogonal Projections](https://lectures.quantecon.org/jl/tools_and_techniques/orth_proj.html)
   -  Notes on  [Numerical Linear Algebra](https://nbviewer.jupyter.org/github/ubcecon/ECON622_2019/blob/master/notebooks/numerical_linear_algebra.ipynb) applying generic programming
7. **September 25th**
   - The Schur Decomposition
   - [Desktop Tools/Package Management](https://lectures.quantecon.org/jl/more_julia/tools_editors.html)
   - [Git and Github](https://lectures.quantecon.org/jl/more_julia/version_control.html)
8. **September 30th**: 
   - [Finite Markov Chains](https://julia.quantecon.org/tools_and_techniques/finite_markov.html)
9. **October 2nd**
   - [Continuous Time Markov Chains](https://nbviewer.jupyter.org/github/ubcecon/ECON622_2019/blob/master/notebooks/numerical_linear_algebra.ipynb) 
   - Self-study [Discrete State Dynamic Programming](https://lectures.quantecon.org/jl/dynamic_programming/discrete_dp.html)   
10. **October 7th**
    - [Testing and Packages](https://julia.quantecon.org/more_julia/testing.html)
    - [Conditioning and Numerical Stability](https://nbviewer.jupyter.org/github/ubcecon/ECON622_2019/blob/master/notebooks/iterative_methods_sparsity.ipynb) 
11. **October 9th**:
    - [Iterative Methods](https://nbviewer.jupyter.org/github/ubcecon/ECON622_2019/blob/master/notebooks/iterative_methods_sparsity.ipynb) 
    - [Intro to AD](https://julia.quantecon.org/more_julia/optimization_solver_packages.html#Introduction-to-Automatic-Differentiation)
12. **October 14th**: NO CLASS (Thanksgiving)
13. **October 16th**: Clusters + Debugging
    - See https://github.com/ubcecon/cluster_tools for instructions on setting up/using the cluster.
    - [Juno debugger](https://docs.junolab.org/latest/man/debugging/)
14. **October 21th**: 
    - [Optimization algorithms](https://schrimpf.github.io/AnimatedOptimization.jl/optimization/)
    - [Optimization packages](https://julia.quantecon.org/more_julia/optimization_solver_packages.html#Optimization)
15. **October 23th**: 
    - [Extremum estimation](https://schrimpf.github.io/GMMInference.jl/extremumEstimation/) and [inference](https://schrimpf.github.io/GMMInference.jl/identificationRobustInference/)    
16. **October 28th**: 
    - [Empirical likelihood](https://schrimpf.github.io/GMMInference.jl/empiricalLikelihood/) 
17. **October 30th**:
    - [Bootstrap](https://schrimpf.github.io/GMMInference.jl/bootstrap/)    
18. **November 4th**:
    - [Coding for performance](https://github.com/schrimpf/ARGridBootstrap)
19. **November 6th**:
    - [GPU usage](https://github.com/schrimpf/ARGridBootstrap)
20. **November 11th**: NO CLASS (Remembrance Day)
21. **November 13th**:    
    <!-- [Structural estimation](http://faculty.arts.ubc.ca/pschrimpf/628/rustrothwell.html) link will change -->
    - [Machine learning](https://schrimpf.github.io/NeuralNetworkEconomics.jl/ml-intro/) or [older version with slides](http://faculty.arts.ubc.ca/pschrimpf/628/machineLearningAndCausalInference.html)
22. **November 18th**:
    - [Machine learning in Julia](https://github.com/schrimpf/NeuralNetworkEconomics.jl)
23. **November 20th**:
    - [Neural Networks](https://github.com/schrimpf/NeuralNetworkEconomics.jl)
    - Good references: [Deep Learning](http://www.deeplearningbook.org/) and [Knet.jl book](https://denizyuret.github.io/Knet.jl/latest/#Textbook-1)
24. **November 25th**:
    - [Convolutional and Recurrent Neural Networks](https://github.com/schrimpf/NeuralNetworkEconomics.jl)
25. **November 27th**: 
    - [Project proposal due](final_project.md)
26. **December 18th**
    - Final Project due December 18th

