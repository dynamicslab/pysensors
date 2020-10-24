---
title: 'PySensors: A Python package for sparse sensor placement'
tags:
  - Python
  - machine learning
authors:
  - name: Brian M. de Silva
    affiliation: 1
  - name: Krithika Manohar
    affiliation: 2
  - name: Emily Clark
    affiliation: 3
  - name: Bingni W. Brunton
    affiliation: 4
  - name: J. Nathan Kutz
    affiliation: 1
  - name: Steven L. Brunton
    affiliation: 2
affiliations:
 - name: Department of Applied Mathematics, University of Washington
   index: 1
 - name: Department of Mechanical Engineering, University of Washington
   index: 2
 - name: Department of Physics, University of Washington
   index: 3
 - name: Department of Biology, University of Washington
   index: 4
date: 24 October 2020
bibliography: paper.bib
---


# Summary

Successful predictive modeling and control of engineering and natural processes is often entirely determined by *in situ* measurements and feedback from sensors [@Brunton2019book]. 
However, deploying sensors into complex environments, including in application areas such as manufacturing [@Manohar2018jms], geophysical environments [@Yildirim:2009], and biological processes [@colvert2017local;@Mohren2018pnas], is often expensive and challenging. 
Furthermore, modeling outcomes are extremely sensitive to the location and number of these sensors, motivating optimization strategies for the principled placement of  sensors for different decision-making tasks. 
In general, choosing the globally optimal placement within the search space of a large-scale complex system is an intractable computation, in which the number of possible placements grows combinatorially with the number of candidates. 
While sensor placement has traditionally been guided by expert knowledge and first principles models, increases in system complexity, emerging sensor technologies, and innovations in data-driven modeling strategies motivates automated algorithms for optimizing sensor placements.

A number of automated sensor placement methods have been developed in recent years, designed to optimize outcomes in the design of experiments [@Boyd2004convexbook;@joshi2008sensor], convex [@joshi2008sensor;@brunton2016sparse] and submodular objective functions [@summers2015submodularity], information theoretic and Bayesian criteria [@Caselton1984spl;@krause2008near;@Lindley1956ams;@Sebastiani2000jrss;@Paninski2005nc], optimal control [@Dhingra2014cdc;@Munz2014ieeetac;@Zare2018arxiv;@Manohar2018arxivB], for sampling and estimating signals over graphs [@Ribeiro2010sigcomm;@DiLorenzo2016ieee;@Chen2016ieee;@Chepuri2016sam], and reduced order modeling [@Barrault2004crm;@willcox2006unsteady;@Chaturantabut2010siamjsc;@Chaturantabut2012siamjna;@drmac2016siam;@manohar2018data;@clark2018greedy].
Maximizing the impact of sensor placement algorithms requires tools to make them accessible to scientists and engineers across various domains and at various levels of mathematical expertise and sophistication.

`PySensors` is a Python package for the scalable optimization of sensor placements from data. In particular, `PySensors` provides tools for sparse sensor placement optimization approaches that employ data-driven dimensionality reduction  [@brunton2016sparse;@manohar2018data]. This approach results in near-optimal placements for various decision-making tasks and can be readily customized using different optimization algorithms and objective functions.

The `PySensors` package can be used by both researchers looking to advance state-of-the-art methods and practitioners seeking simple sparse sensor selection methods for their applications of interest.
Straightforward methods and abundant examples help new users to quickly and efficiently leverage existing methods to their advantage.
At the same time, modular classes leave flexibility for users to experiment with and plug in new sensor selection algorithms or dimensionality reduction techniques.
Users of `scikit-learn` will find `PySensors` objects familiar, intuitive, and compatible with existing `scikit-learn` routines such as cross-validation.


# Features

`PySensors` enables the sparse placement of sensors for two classes of problems: reconstruction and classification.
For reconstruction problems the package implements a unified `SensorSelector` class, with methods for efficiently analyzing the effects that data or sensor quantity have on reconstruction performance [@manohar2018data]. 
Sensor selection is based on the computationally efficient and flexible QR algorithm [@duersch2015true;@martinsson2015blocked;@Martinsson2017siamjsc], which has recently been used for hyper-reduction in reduced-order modeling [@drmac2016siam] and for sparse sensor selection [@manohar2018data]. 
Often different sensor locations impose variable costs, e.g. if measuring sea-surface temperature, it may be more expensive to place buoys/sensors in the middle of the ocean than close to shore.
These costs can be taken into account during sensor selection via a built-in cost-sensitive optimization routine [@clark2018greedy].
For classification tasks, the package implements the Sparse Sensor Placement Optimization for Classification (SSPOC) algorithm [@brunton2016sparse], allowing one to optimize sensor placement for classification accuracy. 
The algorithm is related to compressed sensing optimization [@Candes2006cpam;@Donoho2006ieeetit;@Baraniuk2007ieeespm], but identifies the sparsest set of sensors that reconstructs a discriminating plane in a feature subspace. 
This SSPOC implementation is fully general in the sense that it can be used in conjunction with any linear classifier. 
Additionally, `PySensors` provides methods to enable straightforward exploration of the impacts of primary hyperparameters like the number of sensors or basis modes.

It is well known [@manohar2018data] that the basis in which one represents measurement data can have a pronounced effect on the sensors that are selected and the quality of the reconstruction.
Users can readily switch between different bases typically employed for sparse sensor selection, including principal component analysis (PCA) modes and random projections.
Because `PySensors` was built with `scikit-learn` compatibility in mind, it is easy to use cross-validation to select among possible choices of bases, basis modes, and other hyper-parameters.

Finally, included with `PySensors` is a large suite of examples, implemented as Jupyter notebooks.
Some of the examples are written in a tutorial format and introduce new users to the objects, methods, and syntax of the package.
Other examples demonstrate intermediate-level concepts such as how to visualize model parameters and performance, how to combine `scikit-learn` and `PySensors` objects, selecting appropriate parameter values via cross-validation, and other best-practices.
Further notebooks use `PySensors` to solve challenging real-world problems.
The notebooks reproduce many of the examples from the papers upon which the package is based [@manohar2018data;@clark2018greedy;@brunton2016sparse].
To help users begin applying `PySensors` to their own datasets even faster, interactive versions of every notebook are available on Binder.
Together with comprehensive documentation, the examples will compress the learning curve of learning a new software package. 

# Acknowledgments
The authors acknowledge support from the Air Force Office of Scientific Research (AFOSR FA9550-19-1-0386) and The Boeing Corporation.  JNK acknowledges support from the Air Force Office of Scientific Research (AFOSR FA9550-19-1-0011)

# References