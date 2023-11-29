# ECON622

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

**Paul**

1. **September 6**: Environment and Introduction to Julia
    - Environment: read one or both of these on your own and install Julia, IJulia, and VSCode, preferrably before the first class
        - [Julia Environment](https://quantecon.github.io/lecture-julia.myst/getting_started_julia/getting_started.html)
        - [MoJuWo: Writing your code](https://modernjuliaworkflows.github.io/pages/writing/)
    - In class: Motivating econometric examples
    - Self-study: [Introductory Examples](https://quantecon.github.io/lecture-julia.myst/getting_started_julia/julia_by_example.html)
2. **September 11**: From Economic Models to Code
   - In class: [Implementing random coefficient demand \& numeric integration methods](https://github.com/ubcecon/ECON622_BLP.jl/blob/main/docs/blp.jmd)
   - Self-study: [Julia Essentials](https://quantecon.github.io/lecture-julia.myst/getting_started_julia/julia_essentials.html)
   - Self-study: [Fundamental Types](https://quantecon.github.io/lecture-julia.myst/getting_started_julia/fundamental_types.html)
3. **September 13**: Optimization
   - In class: discussion of types and using them to [reorganize numeric integration code from last time](https://github.com/ubcecon/ECON622_BLP.jl/blob/main/src/integrate.jl) and [testing them](https://github.com/ubcecon/ECON622_BLP.jl/blob/main/test/runtests.jl)
   - [Optimization algorithms](https://schrimpf.github.io/AnimatedOptimization.jl/optimization/)
   - [Optimization packages](https://quantecon.github.io/lecture-julia.myst/more_julia/optimization_solver_packages.html#Optimization)
   - Self-study: [Generic Programming](https://quantecon.github.io/lecture-julia.myst/more_julia/generic_programming.html)
   - Self-study: [Linear Algebra](https://quantecon.github.io/lecture-julia.myst/tools_and_techniques/linear_algebra.html)
4. **September 18**: Automatic Differentiation
   - In class: Overview of [optimization packages](jmd/opt/optimization_packages.jmd) and [automatic differentiation packages](jmd/opt/autodiff.jmd)
   - Self-study: [Intro to AD](https://quantecon.github.io/lecture-julia.myst/more_julia/optimization_solver_packages.html#Introduction-to-Automatic-Differentiation)
   - Self-study: [Differentiation for Hackers](https://github.com/MikeInnes/diff-zoo)
   - Self-study: [Engineering Trade-Offs in Automatic Differentiation: from TensorFlow and PyTorch to Jax and Julia](http://www.stochasticlifestyle.com/engineering-trade-offs-in-automatic-differentiation-from-tensorflow-and-pytorch-to-jax-and-julia/)
   - Optional:
      - [Understanding automatic differentiation (in Julia)](https://www.youtube.com/watch?v=UqymrMG-Qi4)
      - [Forward and Reverse Automatic Differentiation In A Nutshell](https://rawcdn.githack.com/mitmath/matrixcalc/e90417f46a20bec6d9c743c6b7bf5b178e77913a/automatic_differentiation_done_quick.html)
5. **Sepember 20**: Extremum Estimation
   - [Extremum estimation](https://schrimpf.github.io/GMMInference.jl/extremumEstimation/) and [inference](https://schrimpf.github.io/GMMInference.jl/identificationRobustInference/)
   - [Empirical likelihood](https://schrimpf.github.io/GMMInference.jl/empiricalLikelihood/)
   - [Bootstrap](https://schrimpf.github.io/GMMInference.jl/bootstrap/)
   -  Self-study: [General Packages](https://quantecon.github.io/lecture-julia.myst/more_julia/general_packages.html)
   -  Self-study: [Data and Statistical Packages](https://quantecon.github.io/lecture-julia.myst/more_julia/data_statistical_packages.html)
6. **September 25**: Testing and Package Development
   - In class: development of [our random coefficients demand package as an example](https://github.com/ubcecon/ECON622_BLP.jl)
   - Self-study: [Testing and Packages](https://julia.quantecon.org/software_engineering/testing.html)
   - Self-study: [Git and Github](https://julia.quantecon.org/software_engineering/version_control.html)
7. **September 27**: Code Performance
   - [Coding for performance](https://github.com/schrimpf/ARGridBootstrap) be sure to look at the 2023 branch for the recent additions
   - [GPU usage](https://github.com/schrimpf/ARGridBootstrap)
   - Self-study: [SIMDscan](https://github.com/schrimpf/SIMDscan.jl/): since it briefly came up in class, and I was curious about it, I made a little package for calculating things like cumulative sums and autoregressive simulations using SIMD
   - Self-study: [Need for speed](https://julia.quantecon.org/software_engineering/need_for_speed.html)
   - Self-study: [Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)

**JESSE**

Slides for the lectures can be found [here](https://ubcecon.github.io/ECON622/lectures/index.html)

8. **October 4th**: Factorizations, Direct Methods, and Intro to Regularization
    - **SLIDES**: [Factorizations and Direct Methods](https://ubcecon.github.io/ECON622/lectures/lectures/factorizations_direct_methods.html)
    - Introduction to regularization and implicit bias of algorithms
    - [Numerical Linear Algebra](https://julia.quantecon.org/tools_and_techniques/numerical_linear_algebra.html) applying generic programming
9.  **October 11**: Iterative Methods, Geometry of Optimization, and Rethinking LLS
    - **SLIDES**: [Least Squares](https://ubcecon.github.io/ECON622/lectures/lectures/least_squares.html) and [Iterative Methods](https://ubcecon.github.io/ECON622/lectures/lectures/iterative_methods.html)
    - [Iterative Methods](https://julia.quantecon.org/tools_and_techniques/iterative_methods_sparsity.html)
10. **October 12**: Preconditioning and Overview of Machine Learning
    - **SLIDES**:  [Iterative Methods](https://ubcecon.github.io/ECON622/lectures/lectures/iterative_methods.html) and [Intro to ML](https://ubcecon.github.io/ECON622/lectures/lectures/intro_to_ml.html)
    - Finalize discussion of iterative methods and preconditioning
    - Introduce key concepts about supervised, unsupervised, reinforcement learning, semi-supervised, kernel-methods, deep-learning, etc.
11. **October 16**: Differentiable everything! JAX and Auto-Differentiation/JVP/etc.
    - **SLIDES**: Finish [Intro to ML](https://ubcecon.github.io/ECON622/lectures/lectures/intro_to_ml.html) and start [Differentiation](https://ubcecon.github.io/ECON622/lectures/lectures/differentiation.html)
    - Core JAX transforms
    - Reverse-mode and forward-mode AD.  
    - Jvps and vjps
12. **October 18**: AD of Implicit Functions, Optimization Methods for High-Dimensional Problems
    - **SLIDES**: Finish [Differentiation](https://ubcecon.github.io/ECON622/lectures/lectures/differentiation.html) and start [Optimization](https://ubcecon.github.io/ECON622/lectures/lectures/optimization.html)
    - Implicit differentiation of systems of ODEs, linear systems, etc.
    - Gradient descent variations
13. **October 23th**: Stochastic Optimization Methods
    - **SLIDES**: SGD variations in [Optimization](https://ubcecon.github.io/ECON622/lectures/lectures/optimization.html)
    - SGD and methods for variance reduction in gradient estimates
    - Using SGD-variants in practice within ML pipelines in JAX and Pytorch
14. **October 25th**: Machine Learning Pipelines, HPO, and ERM
    - **SLIDES**: Finished example code of pipelines in [Optimization](https://ubcecon.github.io/ECON622/lectures/lectures/optimization.html), W&B sweeps, and code in `lectures/lectures/examples`
    - **Readings**: [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) Section 5.4 on ERM
15. **October 30**: Neural Networks, Representation Learning, Double-Descent
    - **SLIDES**: [Deep Learning and Representation Learning](https://ubcecon.github.io/ECON622/lectures/lectures/deep_learning.html) and started [Double-Descent and Regularization](https://ubcecon.github.io/ECON622/lectures/lectures/overparameterization.html)
    - **Readings**
      - [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) Section 13.2.1 to 13.2.6 on MLPs and the importance of depth
      - [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) Section 13.5.1 to 13.5.6 on regularization
      - [Mark Schmidt's CPSC440 Notes on Neural Networks](https://www.cs.ubc.ca/~schmidtm/Courses/440-W22/L6.pdf) (see [CPSC340](https://www.cs.ubc.ca/~schmidtm/Courses/340-F22/L32.pdf) lectures for a more basic treatment of these topics)
      - [Mark Schmidt's CPSC440 Notes on Double-Descent Curves](https://www.cs.ubc.ca/~schmidtm/Courses/440-W22/L7.pdf) (see [CPSC340](https://www.cs.ubc.ca/~schmidtm/Courses/340-F22/L32.pdf) lectures for a more basic treatment of these topics)
    - **Optional Extra Material**
      - [Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html) Section 32 on representation learning
16. **November 1** Finish Double-Descent and Intro to Kernel Methods and Gaussian Processes
    - **SLIDES**: [Kernel Methods and Gaussian Processes](https://ubcecon.github.io/ECON622/lectures/lectures/kernel_methods.html) and finish [Double-Descent and Regularization](https://ubcecon.github.io/ECON622/lectures/lectures/overparameterization.html)
    - **Readings**
      - If you didn't do it already, read [Mark Schmidt's CPSC440 Notes on Double-Descent Curves and Overparameterization](https://www.cs.ubc.ca/~schmidtm/Courses/440-W22/L7.pdf) (see [CPSC340](https://www.cs.ubc.ca/~schmidtm/Courses/340-F22/L32.pdf) lectures for a more basic treatment of these topics)
      - [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) Section 17.1 and 17.2 on Kernel methods and Gaussian Processes
      - [CPSC340](https://www.cs.ubc.ca/~schmidtm/Courses/340-F22/L22.pdf) has some notes on the "kernel trick", and you can skip over the details on images.  Also see [more advanced notes](https://www.cs.ubc.ca/~schmidtm/Courses/5XX-S22/S8.5.pdf) on kernel methods
      - Finally, your problem set will involve running some simple Gaussian Processes with [GPyTorch](https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html), which will become easier to understand after seeing the theory.
      -  [Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html) Section 18.1 to 18.3 on GPs and kernels
      - Researchers working in GPs love the online textbook [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/chapters/), so you may want to read the intro section on [GP Regression](https://gaussianprocess.org/gpml/chapters/RW2.pdf)

17. **November 6** GUEST LECTURE [GEOFF PLEISS](https://geoffpleiss.com/) FROM UBC STATS

**PHIL**

<!--
Probabilistic Programming Languages (PPLs), Bayesian methods, and intro to generative models
    - Sampling and approximation of model posteriors
    - High-dimensional sampling of latents in practice
    - Variational inference with AD vs. HMC
    NLP, Attention/Transformers, Tuning Foundation Models
-->

18. **November 8** Graphical models introduction
    - **Notes** [Course notes on graphical models](https://raw.githubusercontent.com/ubcecon/ECON622/master/phil_notes/GM_notes.pdf)
    - **Readings** [Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html) Chapter 4.1-4.5
20. **November 13** Fall break
21. **November 15** Fall break
22. **November 20** Using graphical models, Identification, Variable elimination, Undirected graphical models
    - **Notes** [Course notes on graphical models](https://raw.githubusercontent.com/ubcecon/ECON622/master/phil_notes/GM_notes.pdf)
    - **Slides**: [First set of slides](https://ubcecon.github.io/ECON526/lectures/lectures/using_dags.html#/factorizing-the-joint-distribution), [Second set](http://ai.stanford.edu/~paskin/gm-short-course/lec2.pdf)
    - **Readings**:
        - [Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html) Chapter 4.1-4.5
        - [Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html) Chapter 9.5 (Variable elimination)
23. **November 22** Sampling; Inverse CDF, Rejection/importance sampling, MCMC/Metropolis-Hastings algorithm
    - **Notes** [Course notes on Monte-Carlo methods](https://raw.githubusercontent.com/ubcecon/ECON622/master/phil_notes/MCMC_notes.pdf)
    - **Readings**
        - [Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html) Chapter 11
24. **November 27** Local (Gibbs) sampling, Hamiltonian Monte-Carlo, ODE solvers, Probabilistic programming languages (PPLs)
    - **Notes** [Course notes on Monte-Carlo methods](https://raw.githubusercontent.com/ubcecon/ECON622/master/phil_notes/MCMC_notes.pdf)
    - **Readings**
        - [Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html) Chapter 12
        - Hidden Markov Models in [Turing.jl](https://turing.ml/dev/tutorials/04-hidden-markov-model) and [Pyro](https://pyro.ai/examples/hmm.html)
25. **November 29** NLP (short), Tuning foundation models, Graph theory, Graph spectrum, Network games
    - **Readings** 
      - [Course notes on network games](https://raw.githubusercontent.com/ubcecon/ECON622/master/phil_notes/graph_theory_notes.pdf)
      - [Datascience notebook on social & economic networks](https://datascience.quantecon.org/applications/networks.html)
      - [Huggingface fine-tuning demo](https://huggingface.co/docs/transformers/training)
      - Melissa Dell's packages [LinkTransformer](https://github.com/dell-research-harvard/linktransformer) (for merging datasets) and [LayoutParser](https://github.com/Layout-Parser/layout-parser) (for OCR/digitization)
26. **December 4** Student presentations 1
    - Eileen Bian - Application of linear programming
    - Xilin Song - Extension of BLP with product space similarities
    - Jacky Tan - Agent-based models of market microstructure
27. **December 6** Student presentations 2
    - Yidan Chen - TBA
    - Kisho Hoshi - Replication of discrete choice paper with recommender systems
    - Bruno Esposito - Optimizing the code from [Balke and Lamadon (2022)](https://www.aeaweb.org/articles?id=10.1257/aer.20161622) using JAX
    - James Yu - Within-tier heterogeneity in directed search
28. **Decemeber 20**
    - Final Project due


Look under "Releases" for earlier versions of the course.
