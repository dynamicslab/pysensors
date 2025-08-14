---
title: 'PySensors: A Python package for sparse sensor placement'
tags:
  - Python
  - machine learning
authors:
  - name: Brian M. de Silva
    affiliation: 1
    orcid: 0000-0003-0944-900X
  - name: Krithika Manohar
    affiliation: 2
    orcid: 0000-0002-1582-6767
  - name: Emily Clark
    affiliation: 3
  - name: Bingni W. Brunton
    affiliation: 4
    orcid: 0000-0002-4831-3466
  - name: Steven L. Brunton
    affiliation: 2
    orcid: 0000-0002-6565-5118
  - name: J. Nathan Kutz
    affiliation: 1
    orcid: 0000-0002-6004-2275
affiliations:
 - name: Department of Applied Mathematics, University of Washington
   index: 1
 - name: Department of Mechanical Engineering, University of Washington
   index: 2
 - name: Department of Physics, University of Washington
   index: 3
 - name: Department of Biology, University of Washington
   index: 4
date: 13 February 2021
bibliography: paper.bib
---


# Summary

Successful predictive modeling and control of engineering and natural processes is often entirely determined by *in situ* measurements and feedback from sensors [@Brunton2019book], which provide measurements of the state of these processes at specific points in space and time.
However, deploying sensors into complex environments, including in application areas such as manufacturing [@Manohar2018jms], geophysical environments [@Yildirim:2009], and biological processes [@colvert2017local;@Mohren2018pnas], is often expensive and challenging. 
Furthermore, modeling outcomes are extremely sensitive to the location and number of these sensors, motivating optimization strategies for the principled placement of sensors for different decision-making tasks.
In general, choosing the globally optimal placement within the search space of a large-scale complex system is an intractable computation, in which the number of possible placements grows combinatorially with the number of candidates [@ko1995exact]. 
While sensor placement has traditionally been guided by expert knowledge and first principles models, increases in system complexity, emerging sensor technologies, and innovations in data-driven modeling strategies motivates automated algorithms for optimizing sensor placements.

`PySensors` is a Python package for the scalable optimization of sensor placement from data. In particular, `PySensors` provides tools for sparse sensor placement optimization approaches that employ data-driven dimensionality reduction  [@brunton2016sparse;@manohar2018data]. This approach results in near-optimal placements for various decision-making tasks and can be readily customized using different optimization algorithms and objective functions.

The `PySensors` package can be used by both researchers looking to advance state-of-the-art methods and practitioners seeking simple sparse sensor selection methods for their applications of interest.
Straightforward methods and abundant examples help new users to quickly and efficiently leverage existing methods to their advantage.
At the same time, modular classes leave flexibility for users to experiment with and plug in new sensor selection algorithms or dimensionality reduction techniques.
Users of `scikit-learn` will find `PySensors` objects familiar, intuitive, and compatible with existing `scikit-learn` routines such as cross-validation [@scikit-learn].

# Statement of need
Maximizing the impact of sensor placement algorithms requires tools to make them accessible to scientists and engineers across various domains and at various levels of mathematical expertise and sophistication. `PySensors` unifies the algorithms developed in the papers [@manohar2018data;@clark2018greedy;@brunton2016sparse] and their accompanying codes `SSPOR_pub` and `SSPOC_pub` into one software package. The only other packages in this domain of which we are aware are `Chama` [@klise2017sensor] and `Polire` [@narayanan2020toolkit]. While these packages and `PySensors` all enable sparse sensor placement optimization, `Chama` and `Polire` are geared towards event detection and Gaussian processes respectively, whereas `PySensors` is aimed at signal reconstruction and classification tasks.
As such, there are marked differences in the objective functions optimized by `PySensors` and its precursors.
In addition to these two packages, researchers and practitioners have made available various custom scripts for sensor placement. 
Currently, researchers seeking to employ modern sensor placement methods must choose between implementing them from scratch or manually augmenting existing unpolished codes.

Reconstruction and classification tasks often arise in the modeling, prediction, and control of complex processes in geophysics, fluid dynamics, biology, and manufacturing. 
The goal of _reconstruction_ is to recover a high-dimensional signal $\mathbf{x}\in\mathbb{R}^N$ from a limited number of $p$ measurements $y_ i = \mathbf{c}_ i^\top \mathbf{x}$, where each $\mathbf{c}_ i \in \mathbb{R}^N$ represents the action of a sensor. For example, $\mathbf{c}_ i^\top = [1, 0, 0, \dots, 0]$ represents a sensor which takes a point measurement of the first dimension of the signal $\mathbf{x}$.  `PySensors` selects a set of $p$ sensors out of $N$ candidates $\mathbf{c}_ i^\top$ (rows of a measurement matrix $\mathbf{C}:\mathbf{y} = \mathbf{Cx}$) that minimize reconstruction error in a data-dependent basis $\mathbf{\Phi}\in\mathbb{R}^{N\times r}$
$$  \mathbf{C}^ \star=  \underset{\mathbf{C}\in\mathbb{R}^{p\times N}}{\arg\min} \|\mathbf{x} - \mathbf{\Phi}(\mathbf{C\Phi})^{\dagger} \mathbf{y}\|_ 2^2, $$
where $\dagger$ denotes the Moore-Penrose pseudoinverse. The key innovation is to recover the low-dimensional representation $\mathbf{x}_ r \in \mathbb{R}^r$ satisfying $\mathbf{x} = \mathbf{\Phi x}_ r$ via the reconstruction map $\mathbf{\Phi}(\mathbf{C\Phi})^{\dagger}$, ultimately reducing sensor placement to a highly efficient matrix pivoting operation [@manohar2018data]. Similarly, sensor placement for _classification_ [@brunton2016sparse] optimizes the sparsest vector $\mathbf{s}^ \star$ that reconstructs $\mathbf{w}: \mathbf{\Phi}^\dagger\mathbf{s} = \mathbf{w}$ in the low-dimensional feature space, where $\mathbf{w}$ is the the set of weights learned by a linear classifier fit to $\mathbf{x}_ r$.
In this case, the optimal sensor locations are determined by the nonzero components of $\mathbf{s}^ \star$.

The basis $\mathbf{\Phi}$ is explicitly computed from the data using powerful dimensionality reduction techniques such as principal components analysis (PCA) and random projections, which enable significant compression of most signals to $r\ll N$ dimensions. PCA extracts the dominant spatial correlations or _principal components_, the leading eigenvectors of the data covariance matrix. It is computed using the matrix singular value decomposition (SVD) and is closely related to proper orthogonal decomposition (POD); POD modes and principal components are equivalent. 
Other basis choices are possible, such as dynamic mode decomposition for extracting temporally correlated features [@manohar2019optimized].




# Features

`PySensors` enables the sparse placement of sensors for two classes of problems: reconstruction and classification.
For reconstruction problems the package implements a unified `SensorSelector` class, with methods for efficiently analyzing the effects that data or sensor quantity have on reconstruction performance [@manohar2018data]. 
Sensor selection is based on the computationally efficient QR algorithm.
Often different sensor locations impose variable costs, e.g. if measuring sea-surface temperature, it may be more expensive to place buoys/sensors in the middle of the ocean than close to shore.
These costs can be taken into account during sensor selection via a built-in cost-sensitive optimization routine [@clark2018greedy].
For classification tasks, the package implements the Sparse Sensor Placement Optimization for Classification (SSPOC) algorithm [@brunton2016sparse], allowing one to optimize sensor placement for classification accuracy. 
The algorithm is related to compressed sensing optimization [@Candes2006cpam;@Donoho2006ieeetit;@Baraniuk2007ieeespm], but identifies the sparsest set of sensors that reconstructs a discriminating plane in a feature subspace. 
This SSPOC implementation is fully general in the sense that it can be used in conjunction with any linear classifier. 
Additionally, `PySensors` provides methods to enable straightforward exploration of the impacts of primary hyperparameters like the number of sensors or basis modes.

It is well known [@manohar2018data] that the basis in which one represents measurement data can have a pronounced effect on the sensors that are selected and the quality of the reconstruction. 
Users can readily switch between different bases typically employed for sparse sensor selection, including POD modes and random projections.
Because `PySensors` was built with `scikit-learn` compatibility in mind, it is easy to use cross-validation to select among possible choices of bases, basis modes, and other hyperparameters.

Finally, included with `PySensors` is a large suite of examples, implemented as Jupyter notebooks.
Some of the examples are written in a tutorial format and introduce new users to the objects, methods, and syntax of the package.
Other examples demonstrate intermediate-level concepts such as how to visualize model parameters and performance, how to combine `scikit-learn` and `PySensors` objects, selecting appropriate parameter values via cross-validation, and other best-practices.
Further notebooks use `PySensors` to solve challenging real-world problems.
The notebooks reproduce many of the examples from the papers upon which the package is based [@manohar2018data;@clark2018greedy;@brunton2016sparse].
To help users begin applying `PySensors` to their own datasets even faster, interactive versions of every notebook are available on Binder.
Together with comprehensive documentation, the examples will compress the learning curve of learning a new software package. 

# Acknowledgments
The authors acknowledge support from the Air Force Office of Scientific Research (AFOSR FA9550-19-1-0386) and The Boeing Corporation. The work of KM is supported by the National Science Foundation Mathematical Sciences Postdoctoral Research Fellowship (award 1803663). JNK acknowledges support from the Air Force Office of Scientific Research (AFOSR FA9550-19-1-0011)

# References