# ECON622 - Fall 2019

## Computational Economics with Data Science Applications



- **Instructors:** jesse.perla@ubc.ca and schrimpf@mail.ubc.ca
- **Office Hours:** TBD
- **Teaching Assistant:** Mahdi Ebrahimi Kahou, mahdi.ebk@gmail.com
- **Start Time** 8:40am sharp so there are no excuses, please be on time

**Course Description**
This is a graduate topics course in computational economics.


A key purpose of this class is to teach specific techniques,
algorithms, and tools to ensure that students write robust, correct,
and tested code - and hopefully open the research opportunities for
students to move to the cutting edge of quantitative economics.
Beyond the necessary algorithms and new programming languages, another
goal is to ensure that economists are using modern software
engineering tools to allow collaboration - as most projects involve
multiple coauthors and research assistants.  Finally, all of the
practice in this class will be done with the goal of showing how code
used in research can be shared as open-source with the economics
research community (and the scientific computing community as a
whole).

**Grading**
The only way to learn how to apply new programming languages and
methods to economic problems is practice.  To aid in this, a
significant portion of the grade will be regular problem sets.  The
remainder of the grade will be a computational project.


- Nearly weekly problem sets: 50%
- Final Project: 45%
- Participation: 5% (incentive to set alarm)

While the problem sets will be frequent, many will be short to force practice - and hence they will not be weighting uniformly in the grade.  Assume you will get the full participation mark if you rarely miss class.

The final project topics are very open, and the main criteria is that you either (1) learn/use/apply a computational tool to a research topic of your interest or (2) contribute to an open-source computational economics project as a public good.

## Course Parts 

The course will be taught in 4 parts
1. Introduction to Julia and scientific computing (Jesse Perla)
2. Dynamic Programming applications (Jesse Perla)
3. Structural estimation (Paul Schrimpf)
4. Data science and machine learning (Paul Schrimpf)

Throughout the course we will be using the programming language Julia.
     
## Topics

### Jesse

1. Introduction to Julia 
   - Learning the Julia programming language, with simple applications
   - Generic and Functional programming, multiple dispatch
  
2. Software engineering tools: source-code control, unit testing, and continuous integration
   - Git and Github version tracking, diffs, collaboration, Pull Requests, etc.
    - Reproducible environments: package managers, and virtual environments
    - Unit and regression testing frameworks, benchmarking, and continuous-integration
    - Using a cluster/parallel programming
    
3. Function approximation and quadrature for solving dynamic economic models
    - Interpolation
    - Quadrature
    - Spectral methods and functional equations
4. Sparsity/AD/scaling to large problems
   - Application: multiway fixed-effect models
   - Sparse matrices, banded matrices, etc. in Julia
    - Forward/reverse/etc. auto-differentiation
    - Iterative and matrix-free solutions to linear systems, eigensystems, and sparse linear least squares
6. Computational dynamic programming and heterogenous agents

7. Perturbation approaches to solving models with heterogeneous agents
    - Reiter (2009) or similar methods for solving models with aggregate shocks
    
### Paul

1. Extremum estimators & optimization
     - Review of extremum estimators 
     - Introduction to optimization algorithms
     - Inference for extremum estimators
       
2. Unobserved heterogeneity and simulation based inference
     - Numeric integration

3.  Dynamic structural models 
     - Dynamic discrete choice, dynamic games

4.  Machine learning methods 
     
5.  Bayesian methods

Although all these topics are related to econometrics, we will focus
on computational details and not econometric theory. Through example
applications, we will also cover data cleaning, visualization, text
processing, and spatial data. 

