# ECON622 - Fall 2021

This is a graduate topics course in computational economics, with applications in datascience and machine learning.

# Course materials
- Get a [GitHub](www.github.com) ID and apply for the [Student Developer Pack](https://education.github.com/pack) to get further free features
- Consider clicking `Watch` at the top of this repository to see file changes

<!-- ## Accessing the VSE syzygy JupyterHub -->
<!-- 1.  Login to https://vse.syzygy.ca/ with your CWL to ensure you can access our JupyterHub -->
<!-- 2.  Click [Here](https://vse.syzygy.ca/jupyter/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FQuantEcon%2Fquantecon-notebooks-julia&urlpath=lab%2Ftree%2Fquantecon-notebooks-julia) to install the QuantEcon Julia Lectures there -->
<!--     - Later you will need to do a local installation by following the [Getting Started](https://lectures.quantecon.org/jl/getting_started_julia/getting_started.html) but this is a better way to begin -->
<!--     - For support with vse.syzygy.ca, email me@arnavsood.com -->
<!-- 3. To automatically launch the QuantEcon lecture notes on vse.syzygy.ca -->
<!--     - Open the lecture notes in a website (e.g. go to  [Introductory Examples](https://lectures.quantecon.org/jl/getting_started_julia/julia_by_example.html)) -->
<!--     - Hover your mouse over the button "jupyter notebook | run" at the top -->
<!--     - When it pops up a configuration, choose `vse.syzygy.ca (UBC Only)` from the list, move your mouse to somewhere else on the screen -->
<!--     - Now when you click on the "jupyter notebook | run" on any of the Julia lectures (no need to hover again), it will launch in our hub. -->
<!-- 4. Download the extra notebooks from this repository with  [Here](https://vse.syzygy.ca/jupyter/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fubcecon%2FECON622_2019&urlpath=lab%2Ftree%2FECON622_2019%2F) -->
<!--     - To update this repository when we create new notebooks, just click on that link again to clone. -->

<!-- In all cases, the reset a notebook, delete it and click on the launch of clone links again. -->

<!-- Most of the course will be taught using Julia, but we will briefly introduce Python (or R) for discussing topics where Julia is not ideal. -->

## Syllabus
See [Syllabus](syllabus.md) for more details


## Problem Sets

See [problemsets.md](problemsets.md).



## Lectures


1. **January 14**: Environment and Introduction to Julia
    - [Julia Environment](https://quantecon.github.io/lecture-julia.myst/getting_started_julia/getting_started.html), [Introductory Examples](https://quantecon.github.io/lecture-julia.myst/getting_started_julia/julia_by_example.html)

2. **January 14**: Introduction and Variations on Fixed-points
   - [Introductory Examples](https://quantecon.github.io/lecture-julia.myst/getting_started_julia/julia_by_example.html)
3. **January 21**: Introduction to types
   -  Self-study: [Julia Essentials](https://quantecon.github.io/lecture-julia.myst/getting_started_julia/julia_essentials.html)
   -  Self-study: [Fundamental Types](https://quantecon.github.io/lecture-julia.myst/getting_started_julia/fundamental_types.html)
   -  Start [Intro to Generic Programming](https://quantecon.github.io/lecture-julia.myst/getting_started_julia/introduction_to_types.html)
4. **January 21**
   -  Self-study: [Intro to Generic Programming](https://quantecon.github.io/lecture-julia.myst/getting_started_julia/introduction_to_types.html)
   -  Self-study: [Generic Programming](https://quantecon.github.io/lecture-julia.myst/more_julia/generic_programming.html)
5. **January 28**
   -  [Generic Programming](https://quantecon.github.io/lecture-julia.myst/more_julia/generic_programming.html)
   -  Self-study: [General Packages](https://quantecon.github.io/lecture-julia.myst/more_julia/general_packages.html)
   -  Self-study: [Data and Statistical Packages](https://quantecon.github.io/lecture-julia.myst/more_julia/data_statistical_packages.html)
   -  Notes on [Quadrature](https://nbviewer.jupyter.org/github/ubcecon/ECON622_2019/blob/master/notebooks/quadrature.ipynb) applying generic programming
6. **January 28**
   -  Self-study: [Linear Algebra](https://quantecon.github.io/lecture-julia.myst/tools_and_techniques/linear_algebra.html)
   -  Self-study: [Orthogonal Projections](https://quantecon.github.io/lecture-julia.myst/tools_and_techniques/orth_proj.html)
   -  Notes on  [Numerical Linear Algebra](https://nbviewer.jupyter.org/github/ubcecon/ECON622_2019/blob/master/notebooks/numerical_linear_algebra.ipynb) applying generic programming
7. **February 4**
   - The Schur Decomposition
   - [Desktop Tools/Package Management](https://quantecon.github.io/lecture-julia.myst/more_julia/tools_editors.html)
   - [Git and Github](https://quantecon.github.io/lecture-julia.myst/more_julia/version_control.html)
8. **February 4**:
   - [Finite Markov Chains](https://quantecon.github.io/lecture-julia.myst/tools_and_techniques/finite_markov.html)
9. **February 11**
   - [Continuous Time Markov Chains](https://nbviewer.jupyter.org/github/ubcecon/ECON622_2019/blob/master/notebooks/numerical_linear_algebra.ipynb)
   - Self-study [Discrete State Dynamic Programming](https://quantecon.github.io/lecture-julia.myst/dynamic_programming/discrete_dp.html)
10. **February 11**
    - [Testing and Packages](https://quantecon.github.io/lecture-julia.myst/more_julia/testing.html)
    - [Conditioning and Numerical Stability](https://nbviewer.jupyter.org/github/ubcecon/ECON622_2019/blob/master/notebooks/iterative_methods_sparsity.ipynb)
11. **February 18**
    - [Iterative Methods](https://nbviewer.jupyter.org/github/ubcecon/ECON622_2019/blob/master/notebooks/iterative_methods_sparsity.ipynb)
    - [Intro to AD](https://quantecon.github.io/lecture-julia.myst/more_julia/optimization_solver_packages.html#Introduction-to-Automatic-Differentiation)
    - Self-study: https://github.com/ubcecon/cluster_tools for instructions on setting up/using the cluster.
12. **February 18**
    - [Optimization algorithms](https://schrimpf.github.io/AnimatedOptimization.jl/optimization/)
    - [Optimization packages](https://quantecon.github.io/lecture-julia.myst/more_julia/optimization_solver_packages.html#Optimization)
15. **March 4**
    - [Extremum estimation](https://schrimpf.github.io/GMMInference.jl/extremumEstimation/) and [inference](https://schrimpf.github.io/GMMInference.jl/identificationRobustInference/)
16. **March 4**
    - [Empirical likelihood](https://schrimpf.github.io/GMMInference.jl/empiricalLikelihood/)
17. **March 11**
    - [Bootstrap](https://schrimpf.github.io/GMMInference.jl/bootstrap/)
18. **March 11**:
    - [Coding for performance](https://github.com/schrimpf/ARGridBootstrap)
    - Self-study: [Need for speed](https://quantecon.github.io/lecture-julia.myst/more_julia/need_for_speed.html)
19. **March 18**
    - [GPU usage](https://github.com/schrimpf/ARGridBootstrap)
21. **March 18**
    <!-- [Structural estimation](http://faculty.arts.ubc.ca/pschrimpf/628/rustrothwell.html) link will change -->
    - [Machine learning](https://schrimpf.github.io/NeuralNetworkEconomics.jl/ml-intro/) or [older version with slides](http://faculty.arts.ubc.ca/pschrimpf/628/machineLearningAndCausalInference.html)
22. **March 25**
    - [Machine learning in Julia](https://github.com/schrimpf/NeuralNetworkEconomics.jl)
23. **March 25**
    - [Neural Networks](https://github.com/schrimpf/NeuralNetworkEconomics.jl)
24. **April 1**
    - [Convolutional and Recurrent Neural Networks](https://github.com/schrimpf/NeuralNetworkEconomics.jl)
25. **April 8**
    - [Project proposal due](final_project.md)
27. **April 22**
    - Final Project due


Look under "Releases" for earlier versions of the course.
