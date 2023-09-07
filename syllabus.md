# ECON622 - Fall 2023

## Computational Economics with Data Science Applications

- **Instructors:**
  - Paul Schrimpf, schrimpf@mail.ubc.ca
  - Jesse Perla, jesse.perla@ubc.cas
  - Phil Solimine,
- **Office Hours:**
- **Class Time** Mondays & Wednesdays 9:30-11:00 in [Buchanon D301](https://learningspaces.ubc.ca/classrooms/buch-d301)


Classes will held in-person.
<!-- In-person attendance is preferred, but classes will be streamed and recorded as well. -->


## Course Description

This is a graduate topics course in computational economics.  We intend
this to be useful for a large number of fields, but it is most useful
for anyone likely to:
1. Estimate a structural model
2. Solve a dynamic
model
3. Collect and use data beyond what is possible in Stata (e.g medium/big data, textual data, etc.); or
4. Implement econometric techniques that go beyond what is available in Stata
5. Understand how new ML techniques can be applied to economics

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


- Nearly weekly problem sets: 40%
- Final Project: 40%
- Presentation: 15%
- Participation: 5%

While the problem sets will be frequent, many will be short to force
practice (and will not be weighed identically) Assume you will get the
full participation mark if you rarely miss class.

The final project topics are very open, and the main criteria is that
you either (1) learn/use/apply a computational tool to a research
topic of your interest or (2) contribute to an open-source
computational economics project as a public good.

There will be short presentations in the last week of class. The topic
of the presentation is flexible. It should be about computation and
economics. It may be related to your final project. For example, you
could present a summary of your plan for your project and any
difficulties encountered so far. It could be about someone else's
paper on a technique that might be useful for your project.

<!-- ## Course Parts -->

<!-- The course will be taught in 3 parts by  -->
<!-- 1. Introduction to Julia and scientific computing -->
<!-- 2. Dynamic Programming applications -->
<!-- 3. Structural estimation -->
<!-- 4. Data science and machine learning -->

<!-- Throughout the course we will be using the programming language Julia. -->

<!-- The course will be held in-person in Iona 633. Lectures will also be recorded and streamed, see canvas for links. -->

## Topics

The course will be taught in 3 parts, one with each instructor.

We may not cover all these topics. A tentative schedule, based on last
year's course is availabe on the course webpage.

### Paul

This part of the course will introduce Julia and illustrate how it can be used for econometrics, especially structural estimation.

1. Introduction to Julia
   - Learning the Julia programming language, with simple applications
   - Generic and Functional programming, multiple dispatch

2. Software engineering tools: source-code control, unit testing, and continuous integration
   - Git and Github version tracking, diffs, collaboration, Pull Requests, etc.
   - Reproducible environments: package managers, and virtual environments
   - Unit and regression testing frameworks, benchmarking, and continuous-integration

3. Extremum estimators & optimization
     - Review of extremum estimators
     - Introduction to optimization algorithms
     - Automatic Differentiation
     - Inference for extremum estimators

### Jesse

This section will concentration on machine learning and deep learning techniques, and built computational tools such as working with gradients.  Much of the code with be introduced using python toolkits such as JAX and PyTorch


1. Iterative and matrix-free methods, pre-conditioning and regularization
2. Introduction to Pytorch, JAX, and "ML Devops"
3. Reverse-mode and forward-mode AD.  Differentiable everything!
4. Probabilistic Programming Languages (PPLs), Bayesian methods, and intro to generative models
5. Gaussian Processes and Intro to Bayesian Optimization
6. Neural Networks and Function Approximation
7. Intro to Neural Networks, Function Approximation, and Representation Learning
8. Deep Learning and Dynamic Models
9. Double-descent, regularization, and generalization

## Phil

See schedule on [README.md](README.md).


## UBC values and policies

UBC provides resources to support student learning and to maintain
healthy lifestyles but recognizes that sometimes crises arise and so
there are additional resources to access including those for survivors
of sexual violence. UBC values respect for the person and ideas of all
members of the academic community. Harassment and discrimination are
not tolerated nor is suppression of academic freedom. UBC provides
appropriate accommodation for students with disabilities and for
religious and cultural observances. UBC values academic honesty and
students are expected to acknowledge the ideas generated by others and
to uphold the highest academic standards in all of their
actions. Details of the policies and how to access support are
available here
[https://senate.ubc.ca/policiesresources-support-student-success](https://senate.ubc.ca/policiesresources-support-student-success)
