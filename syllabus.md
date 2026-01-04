# ECON622 - Winter 2026

## Computational Economics with Data Science Applications

- **Instructors:**
  - Paul Schrimpf, schrimpf@mail.ubc.ca
  - Jesse Perla, jesse.perla@ubc.cas
- **Office Hours:**
- **Class Time** Mondays & Wednesdays 12:30-2:00 in Iona 533


Classes will held in-person.

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

Students may work together on assignments, but each student should write their own answers. If you work closely together with someone and consequently have very similar code, you should state with whom you worked on your assignment.

The use of generative AI tools such as GitHub Copilot or ChatGPT is allowed as long as you disclose their use.

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

The course will be taught in 2 parts, one with each instructor.

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


# Policies on AI and Collaboration

## Policy on the Use of AI Learning Tools:

Students are permitted to use artificial intelligence tools, including generative AI, to gather information, review concepts or to help produce assignments. However, students are ultimately accountable for the work they submit, and any content generated or supported by an artificial intelligence tool must be cited appropriately. Use of AI tools of any type is not permitted during exams.


## Policy on Collaborating on Assignments:
Students are encouraged to work together on assignments throughout the course.  However, all students must independently write solutions up and submit separately.  If the code and writeup is too close then a penalty or a zero will be given for all students involved.

# Concessions

## Policy on Academic Concessions

There are only three acceptable grounds for academic concessions at UBC: unexpected changes in personal responsibilities that create a schedule conflict; medical circumstances; and compassionate grounds when the student experiences a traumatic event, sexual assault, or death in the family or of a close friend. Academic concessions for graded work and exams are granted for work that will be missed due to unexpected situations or circumstances. Situations that are expected (such as time constraints due to workload in other courses) or are predictable (such as being scheduled for paid work) are not grounds for academic concession.  See https://www.arts.ubc.ca/degree-planning/academic-performance/academic-concession/

Requests for academic concessions should be made before the due date for that graded work and/or the writing of the exam. UBC policy does not allow for concessions to students who have missed work because they have registered for a course after the due date for that work. You can read more about the rules for academic concessions here: https://students.ubc.ca/enrolment/academic-learning-resources/academic-concessions. Students in the Faculty of Arts who require a concession can apply for concessions using this form here: https://students.air.arts.ubc.ca/academic-concession-form/. Students in other Faculties should consult their faculty website on academic concessions. Please note that the role of the faculty advising office is to review the evidence and to either support or not support concession requests. The final decision to grant the request always rests with your instructor.

Below are more specific policies for this course


## Problem Set Policy
Email the instructor if something comes up which makes it impossible to complete your problem sets in time.  This can include both official grounds for concessions as well as issues which are below the threshold for a concession, but might impact the students ability to learn the most from an assignment.
- Extensions are typically granted unless it becomes a pattern.
- Regardless, to encourage students to submit assignments as part of their studying process, even if they are very late, grades for late assignments will always be above zero until the last day of class.


# Conduct and Academic Honesty
## Student Success
UBC provides resources to support student learning and to maintain healthy lifestyles but recognizes that sometimes crises arise and so there are additional resources to access including those for survivors of sexual violence. UBC values respect for the person and ideas of all members of the academic community. Harassment and discrimination are not tolerated nor is suppression of academic freedom. UBC provides appropriate accommodation for students with disabilities and for religious, spiritual and cultural observances. UBC values academic honesty and students are expected to acknowledge the ideas generated by others and to uphold the highest academic standards in all of their actions. Details of the policies and how to access support are available here: https://senate.ubc.ca/policies-resources-support-student-success/.

## Policy on sharing course materials:

 All the materials provided to you as part of this course are protected by copyright. All assignment instructions, quiz questions and answers, discussion questions, announcements, lecture slides, audio/video recordings, Canvas modules, and any other materials provided to you by your instructor or in the textbook are for use only by students enrolled in this course this term. Sharing any of these materials beyond this course, including by posting on file-sharing websites (e.g., CourseHero, Google Docs) is a violation of copyright law and an academic offence. Copying and pasting sentences from the lecture notes or the textbook (e.g., definitions) into for-profit software (e.g., Quizlet) is likewise a violation of copyright law, and an academic offence. Violations of this policy will be treated according to the provisos of the Code of Student Conduct. For further information about copyright law, please refer to (https://copyright.ubc.ca/students/).

## Policy on Academic Honesty:

It is the policy of the Vancouver School of Economics to report all violations of UBC’s standards for academic integrity to the office of the Dean of Arts. All violations of academic integrity standards will result in a grade of zero on the relevant assessment (exam, paper, assignment etc.).  Students who do not have a previous offence may have the option to enter into a diversionary process with the Dean of Arts to resolve their misconduct (https://academicintegrity.ubc.ca/diversionary-process/). Any student who has a previous academic offence will be referred to the President’s Advisory Committee on Student Discipline (PACSD) (https://universitycounsel.ubc.ca/homepage/guides-and-resources/discipline/). PACSD may impose additional penalties including: a transcript notation indicating that the student has committed an academic offence, zero in the course, and/or suspension or expulsion from the University. You are personally responsible for understanding and following the UBC’s policies for academic integrity: https://vancouver.calendar.ubc.ca/campus-wide-policies-and-regulations/academic-honesty-and-standards. A Canvas module has been made available you for this purpose titled “Avoiding Academic Misconduct”. It is your responsibility to read the materials in that module before submitting any work in this course. Speak to your instructor if you have any questions regarding the standard for academic integrity at UBC and/or the VSE polices on academic misconduct.
