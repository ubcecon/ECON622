# ECON622

This is a graduate topics course in computational economics, with applications in datascience and machine learning.

# Course materials
- Get a [GitHub](www.github.com) ID and apply for the [Student Developer Pack](https://education.github.com/pack) to get further free features
- Consider clicking `Watch` at the top of this repository to see file changes

## Syllabus

See [Syllabus](syllabus.md) for more details

## Problem Sets

See [problemsets.md](problemsets.md).

## Lectures

**Paul**

1. **September 4**: Environment and Introduction to Julia
    - [Intro slides](https://ubcecon.github.io/ECON622/paul/intro.html)
    - Environment: read one or both of these on your own and install Julia, IJulia, and VSCode, preferrably before the first class
        - [MoJuWo: Writing your code](https://modernjuliaworkflows.org/writing/)
        - [Julia Environment](https://quantecon.github.io/lecture-julia.myst/getting_started_julia/getting_started.html)
    - In class: Motivating econometric examples
    - Self-study: [Introductory Examples](https://quantecon.github.io/lecture-julia.myst/getting_started_julia/julia_by_example.html) or [Chapter 1 of *Scientific Programming in Julia*](https://juliateachingctu.github.io/Scientific-Programming-in-Julia/stable/lecture_01/motivation/)
2. **September 9**: Integration
   - [Slides](https://ubcecon.github.io/ECON622/paul/integration.html)
   - Self-study: [Julia Essentials](https://quantecon.github.io/lecture-julia.myst/getting_started_julia/julia_essentials.html) and [Fundamental Types](https://quantecon.github.io/lecture-julia.myst/getting_started_julia/fundamental_types.html)
   - Self-study: [Chapter 2 of *Scientific Programming in Julia*](https://juliateachingctu.github.io/Scientific-Programming-in-Julia/stable/lecture_02/lecture/)
3. **September 11**: Nonlinear Equation Solving
   - [Slides](https://ubcecon.github.io/ECON622/paul/equationsolving.html)
    - Self-study: [Design patterns: good practices and structured thinking](https://juliateachingctu.github.io/Scientific-Programming-in-Julia/dev/lecture_03/lecture/)
4. **September 16**: Project Best Practices
   - [Slides](https://ubcecon.github.io/ECON622/paul/bestpractices.html)
   - Self-study: [Package development, unit tests, & CI](https://juliateachingctu.github.io/Scientific-Programming-in-Julia/dev/lecture_04/lecture/)
   - Self-study: [Testing and Packages](https://julia.quantecon.org/software_engineering/testing.html)
   - Self-study: [Git and Github](https://julia.quantecon.org/software_engineering/version_control.html)
5. **September 18**: clean up example project, introduction to automatic differentiation
   - In class: [automatic differentiation packages](qmd/autodiff.qmd), [slides](https://ubcecon.github.io/ECON622/paul/autodiff.html)
   - Self-study: [Automatic Differentation in *Scientific Programming in Julia*](https://juliateachingctu.github.io/Scientific-Programming-in-Julia/dev/lecture_08/lecture/)
   - Self-study: [Differentiation for Hackers](https://github.com/MikeInnes/diff-zoo)
   - Self-study: [Engineering Trade-Offs in Automatic Differentiation: from TensorFlow and PyTorch to Jax and Julia](http://www.stochasticlifestyle.com/engineering-trade-offs-in-automatic-differentiation-from-tensorflow-and-pytorch-to-jax-and-julia/)
   - Optional:
      - [Understanding automatic differentiation (in Julia)](https://www.youtube.com/watch?v=UqymrMG-Qi4)
      - [Forward and Reverse Automatic Differentiation In A Nutshell](https://rawcdn.githack.com/mitmath/matrixcalc/e90417f46a20bec6d9c743c6b7bf5b178e77913a/automatic_differentiation_done_quick.html)
      - [Intro to AD](https://quantecon.github.io/lecture-julia.myst/more_julia/optimization_solver_packages.html#Introduction-to-Automatic-Differentiation)
6. **September 23**: Optimization
   - [optimization](qmd/optimization.qmd), [slides](https://ubcecon.github.io/ECON622/paul/optimization.html)
   - [Optimization algorithms](https://schrimpf.github.io/AnimatedOptimization.jl/optimization/)
   - [Optimization packages](https://quantecon.github.io/lecture-julia.myst/more_julia/optimization_solver_packages.html#Optimization)
7. **September 25**: Extremum Estimation
   - [Extremum estimation](https://schrimpf.github.io/GMMInference.jl/extremumEstimation/) and [inference](https://schrimpf.github.io/GMMInference.jl/identificationRobustInference/)
   - [Empirical likelihood](https://schrimpf.github.io/GMMInference.jl/empiricalLikelihood/)
   - [Bootstrap](https://schrimpf.github.io/GMMInference.jl/bootstrap/)

8. **October 2** Function Approximation
   - [Slides](https://ubcecon.github.io/ECON622/paul/approximation.html#/title-slide)
9. **October 7** Code Performance
   - [Coding for performance](https://github.com/schrimpf/ARGridBootstrap) be sure to look at the 2023 branch for the recent additions
   - [GPU usage](https://github.com/schrimpf/ARGridBootstrap)
   - Self-study: [SIMDscan](https://github.com/schrimpf/SIMDscan.jl/): since it briefly came up in class, and I was curious about it, I made a little package for calculating things like cumulative sums and autoregressive simulations using SIMD
   - Self-study: [Need for speed](https://julia.quantecon.org/software_engineering/need_for_speed.html)
   - Self-study: [Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)
10. **October 9** Dynamic Programming
   - [Slides](qmd/dynamicprogramming.qmd)
11. **October 16** Debiased Machine Learning
   - [My notes](https://schrimpf.github.io/NeuralNetworkEconomics.jl/ml-doubledebiased/)
   - Self-study: [CausalML](https://causalml-book.org/), [Chernozhukov, Newey, Quintas-Martinez, & Syrgkanis (2024)](https://arxiv.org/abs/2104.14737)

**JESSE**

Slides for the lectures can be found [here](https://ubcecon.github.io/ECON622/lectures/index.html)

12. **October 21**: Factorizations, Direct Methods, and Intro to Regularization
    - **SLIDES**: [Factorizations and Direct Methods](https://ubcecon.github.io/ECON622/lectures/lectures/factorizations_direct_methods.html)
    - Introduction to regularization and implicit bias of algorithms
    - [Numerical Linear Algebra](https://julia.quantecon.org/tools_and_techniques/numerical_linear_algebra.html) applying generic programming
13.  **October 23**: Iterative Methods, Geometry of Optimization, Rethinking LLS, and Preconditioning
    - **SLIDES**: [Least Squares](https://ubcecon.github.io/ECON622/lectures/lectures/least_squares.html) and [Iterative Methods](https://ubcecon.github.io/ECON622/lectures/lectures/iterative_methods.html)
    - [Iterative Methods](https://julia.quantecon.org/tools_and_techniques/iterative_methods_sparsity.html)
14. **October 28**: Overview of Machine Learning
    - **SLIDES**:  [Intro to ML](https://ubcecon.github.io/ECON622/lectures/lectures/intro_to_ml.html)
    - Finalize discussion of iterative methods and preconditioning
    - Introduce key concepts about supervised, unsupervised, reinforcement learning, semi-supervised, kernel-methods, deep-learning, etc.
    - Basic introduction to JAX and Python frameworks
15. **October 30**: Differentiable everything! JAX and Auto-Differentiation/JVP/etc.
    - **SLIDES**: [Differentiation](https://ubcecon.github.io/ECON622/lectures/lectures/differentiation.html)
    - Reverse-mode and forward-mode AD.
    - Jvps and vjps
    - Implicit differentiation of systems of ODEs, linear systems, etc.
16. **November 4**: High-dimensional optimization and Stochastic Optimization
    - **SLIDES**: [Optimization](https://ubcecon.github.io/ECON622/lectures/lectures/optimization.html)
    - Gradient descent variations
    - Using unbiased estimates instead of gradients
17. **November 6**: Stochastic Optimization Methods and Machine Learning Pipelines
    - **SLIDES**: SGD variations in [Optimization](https://ubcecon.github.io/ECON622/lectures/lectures/optimization.html)
    - W&B sweeps, and code in `lectures/lectures/examples`- 
    - SGD and methods for variance reduction in gradient estimates
    - Using SGD-variants in practice within ML pipelines in JAX and Pytorch
    - **Readings**: [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) Section 5.4 on ERM
18. **November 18**: Neural Networks, Representation Learning, Double-Descent
    - **SLIDES**: [Deep Learning and Representation Learning](https://ubcecon.github.io/ECON622/lectures/lectures/deep_learning.html) and started [Double-Descent and Regularization](https://ubcecon.github.io/ECON622/lectures/lectures/overparameterization.html)
    - **Readings**
      - [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) Section 13.2.1 to 13.2.6 on MLPs and the importance of depth
      - [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) Section 13.5.1 to 13.5.6 on regularization
      - [Mark Schmidt's CPSC440 Notes on Neural Networks](https://www.cs.ubc.ca/~schmidtm/Courses/440-W22/L6.pdf) (see [CPSC340](https://www.cs.ubc.ca/~schmidtm/Courses/340-F22/L32.pdf) lectures for a more basic treatment of these topics)
      - [Mark Schmidt's CPSC440 Notes on Double-Descent Curves](https://www.cs.ubc.ca/~schmidtm/Courses/440-W22/L7.pdf) (see [CPSC340](https://www.cs.ubc.ca/~schmidtm/Courses/340-F22/L32.pdf) lectures for a more basic treatment of these topics)
    - **Optional Extra Material**
      - [Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html) Section 32 on representation learning
19. **November 20** Finish Double-Descent and Intro to Kernel Methods and Gaussian Processes
    - **SLIDES**: Start [Kernel Methods and Gaussian Processes](https://ubcecon.github.io/ECON622/lectures/lectures/kernel_methods.html) and finish [Double-Descent and Regularization](https://ubcecon.github.io/ECON622/lectures/lectures/overparameterization.html)
    - **Readings**
      - If you didn't do it already, read [Mark Schmidt's CPSC440 Notes on Double-Descent Curves and Overparameterization](https://www.cs.ubc.ca/~schmidtm/Courses/440-W22/L7.pdf) (see [CPSC340](https://www.cs.ubc.ca/~schmidtm/Courses/340-F22/L32.pdf) lectures for a more basic treatment of these topics)
      - [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) Section 17.1 and 17.2 on Kernel methods and Gaussian Processes
      - [CPSC340](https://www.cs.ubc.ca/~schmidtm/Courses/340-F22/L22.pdf) has some notes on the "kernel trick", and you can skip over the details on images.  Also see [more advanced notes](https://www.cs.ubc.ca/~schmidtm/Courses/5XX-S22/S8.5.pdf) on kernel methods
      - Finally, your problem set will involve running some simple Gaussian Processes with [GPyTorch](https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html), which will become easier to understand after seeing the theory.
      -  [Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html) Section 18.1 to 18.3 on GPs and kernels
      - Researchers working in GPs love the online textbook [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/chapters/), so you may want to read the intro section on [GP Regression](https://gaussianprocess.org/gpml/chapters/RW2.pdf)

20. **November 25** Finish Kernel Methods and Review "Distances" in a latent space
    - **SLIDES**: Finish [Kernel Methods and Gaussian Processes](https://ubcecon.github.io/ECON622/lectures/lectures/kernel_methods.html)
21. **November 27** "ML Engineering"
    - See example code, W&B, etc.  Linear regression, NN regression, pytorch, pytorch lightning, CLI, etc.
22. **December 2** LLMs and Embeddings
    - **SLIDES**: [Embeddings, NLP, and LLMs](https://ubcecon.github.io/ECON622/lectures/lectures/embeddings_nlp_llm.html)
23. **December 4** Attention and Transformers + Class Presentations
    - **SLIDES**: [Transformers and Attention](https://ubcecon.github.io/ECON622/lectures/lectures/transformers.html) and
24. **December 20**
    - Final Project due


Look under "Releases" or switch to another branch for earlier versions of the course.
