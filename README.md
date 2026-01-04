# ECON622

This is a graduate topics course in computational economics, with applications in datascience and machine learning.

# Course materials
- See [here](https://jlperla.github.io/grad_econ_ML/#getting-started) for instructions on getting started with Github and Python.
    - Julia instructions are also provided for the second half of the course.
- Note the recommendations on [GitHub Academic](https://jlperla.github.io/grad_econ_ML/index.html#getting-started)

## Syllabus

See [Syllabus](syllabus.md) for more details

## Problem Sets

**Jesse**: see the [schedule](https://jlperla.github.io/grad_econ_ML/pages/schedule.html)
**Paul**: see [problemsets.md](problemsets.md)

## Lectures

**Jesse**
For the first half of the course, see the [schedule](https://jlperla.github.io/grad_econ_ML/pages/schedule.html).

**Paul**

1. **February 23**: Environment and Introduction to Julia
    - [Intro slides](https://ubcecon.github.io/ECON622/paul/intro.html)
    - Environment: read one or both of these on your own and install Julia, IJulia, and VSCode, preferrably before the first class
        - [MoJuWo: Writing your code](https://modernjuliaworkflows.org/writing/)
        - [Julia Environment](https://quantecon.github.io/lecture-julia.myst/getting_started_julia/getting_started.html)
    - In class: Motivating econometric examples
    - Self-study: [Introductory Examples](https://quantecon.github.io/lecture-julia.myst/getting_started_julia/julia_by_example.html) or [Chapter 1 of *Scientific Programming in Julia*](https://juliateachingctu.github.io/Scientific-Programming-in-Julia/stable/lecture_01/motivation/)
2. **February 25**: Integration
   - [Slides](https://ubcecon.github.io/ECON622/paul/integration.html)
   - Self-study: [Julia Essentials](https://quantecon.github.io/lecture-julia.myst/getting_started_julia/julia_essentials.html) and [Fundamental Types](https://quantecon.github.io/lecture-julia.myst/getting_started_julia/fundamental_types.html)
   - Self-study: [Chapter 2 of *Scientific Programming in Julia*](https://juliateachingctu.github.io/Scientific-Programming-in-Julia/stable/lecture_02/lecture/)
3. **March 2**: Nonlinear Equation Solving
   - [Slides](https://ubcecon.github.io/ECON622/paul/equationsolving.html)
    - Self-study: [Design patterns: good practices and structured thinking](https://juliateachingctu.github.io/Scientific-Programming-in-Julia/dev/lecture_03/lecture/)
4. **March 4**: Project Best Practices
   - [Slides](https://ubcecon.github.io/ECON622/paul/bestpractices.html)
   - Self-study: [Package development, unit tests, & CI](https://juliateachingctu.github.io/Scientific-Programming-in-Julia/dev/lecture_04/lecture/)
   - Self-study: [Testing and Packages](https://julia.quantecon.org/software_engineering/testing.html)
   - Self-study: [Git and Github](https://julia.quantecon.org/software_engineering/version_control.html)
5. **March 9**: clean up example project, introduction to automatic differentiation
   - In class: [automatic differentiation packages](qmd/autodiff.qmd), [slides](https://ubcecon.github.io/ECON622/paul/autodiff.html)
   - Self-study: [Automatic Differentation in *Scientific Programming in Julia*](https://juliateachingctu.github.io/Scientific-Programming-in-Julia/dev/lecture_08/lecture/)
   - Self-study: [Differentiation for Hackers](https://github.com/MikeInnes/diff-zoo)
   - Self-study: [Engineering Trade-Offs in Automatic Differentiation: from TensorFlow and PyTorch to Jax and Julia](http://www.stochasticlifestyle.com/engineering-trade-offs-in-automatic-differentiation-from-tensorflow-and-pytorch-to-jax-and-julia/)
   - Optional:
      - [Understanding automatic differentiation (in Julia)](https://www.youtube.com/watch?v=UqymrMG-Qi4)
      - [Forward and Reverse Automatic Differentiation In A Nutshell](https://rawcdn.githack.com/mitmath/matrixcalc/e90417f46a20bec6d9c743c6b7bf5b178e77913a/automatic_differentiation_done_quick.html)
      - [Intro to AD](https://quantecon.github.io/lecture-julia.myst/more_julia/optimization_solver_packages.html#Introduction-to-Automatic-Differentiation)
6. **March 11**: Optimization
   - [optimization](qmd/optimization.qmd), [slides](https://ubcecon.github.io/ECON622/paul/optimization.html)
   - [Optimization algorithms](https://schrimpf.github.io/AnimatedOptimization.jl/optimization/)
   - [Optimization packages](https://quantecon.github.io/lecture-julia.myst/more_julia/optimization_solver_packages.html#Optimization)
7. **March 16**: Extremum Estimation
   - [Extremum estimation](https://schrimpf.github.io/GMMInference.jl/extremumEstimation/) and [inference](https://schrimpf.github.io/GMMInference.jl/identificationRobustInference/)
   - [Empirical likelihood](https://schrimpf.github.io/GMMInference.jl/empiricalLikelihood/)
   - [Bootstrap](https://schrimpf.github.io/GMMInference.jl/bootstrap/)

8. **March 18** Function Approximation
   - [Slides](https://ubcecon.github.io/ECON622/paul/approximation.html#/title-slide)
9. **March 23** Code Performance
   - [Coding for performance](https://github.com/schrimpf/ARGridBootstrap) be sure to look at the 2023 branch for the recent additions
   - [GPU usage](https://github.com/schrimpf/ARGridBootstrap)
   - Self-study: [SIMDscan](https://github.com/schrimpf/SIMDscan.jl/): since it briefly came up in class, and I was curious about it, I made a little package for calculating things like cumulative sums and autoregressive simulations using SIMD
   - Self-study: [Need for speed](https://julia.quantecon.org/software_engineering/need_for_speed.html)
   - Self-study: [Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)
10. **March 25** Dynamic Programming
   - [Slides](qmd/dynamicprogramming.qmd)
11. **March 30** Debiased Machine Learning
   - [My notes](https://schrimpf.github.io/NeuralNetworkEconomics.jl/ml-doubledebiased/)
   - Self-study: [CausalML](https://causalml-book.org/), [Chernozhukov, Newey, Quintas-Martinez, & Syrgkanis (2024)](https://arxiv.org/abs/2104.14737)
24. **April 1** Class Presentations
25. **April 6** STAT HOLIDAY
26. **April 8** Class Presentations

Look under "Releases" or switch to another branch for earlier versions of the course.
