---
title: 'PySensors 2.0: A Python Package for Sparse Sensor Placement'
tags:
  - Python
  - machine learning
authors:
  - name: Niharika Karnik
    affiliation: 1
    orcid: 0000-0002-4259-0294
  - name: Yash Bhangale
    affiliation: 1
    orcid: 0009-0008-4163-6538
  - name: Mohammad G. Abdo
    affiliation: 3
    orcid: 0000-0001-9845-6978
  - name: Andrei A. Klishin
    affiliation: 4
    orcid: 0000-0002-5740-8520
  - name: Joshua J. Cogliati
    affiliation: 3
    orcid: 0000-0003-2471-8095
  - name: Bingni W. Brunton
    affiliation: 5
    orcid: 0000-0002-4831-3466
  - name: J. Nathan Kutz
    affiliation: 2
    orcid: 0000-0002-6004-2275
  - name: Steven L. Brunton
    affiliation: 1
    orcid: 0000-0002-6565-5118
  - name: Krithika Manohar
    affiliation: 1
    orcid: 0000-0002-1582-6767
affiliations:
 - name: Department of Mechanical Engineering, University of Washington
   index: 1
 - name: Department of Applied Mathematics, University of Washington
   index: 2
 - name: Idaho National Laboratory
   index: 3
 - name: Department of Mechanical Engineering, University of Hawai'i at Mānoa
   index: 4
 - name: Department of Biology, University of Washington
   index: 5
date: 12 August 2025
bibliography: paper.bib
---


# Summary

`PySensors` is a Python package for selecting and placing a sparse set of sensors for reconstruction and classification tasks. In this major update to `PySensors`, we introduce spatially constrained sensor placement capabilities, allowing users to enforce constraints such as maximum or exact sensor counts in specific regions, incorporate predetermined sensor locations, and maintain minimum distances between sensors. We extend functionality to support custom basis inputs, enabling integration of any data-driven or spectral basis. We also propose a thermodynamic approach that goes beyond a single ``optimal'' sensor configuration and maps the complete landscape of sensor interactions induced by the training data. This comprehensive view facilitates integration with external selection criteria and enables assessment of sensor replacement impacts. The new optimization technique also accounts for over- and under-sampling of sensors, utilizing a regularized least squares approach for robust reconstruction. Additionally, we incorporate noise-induced uncertainty quantification of the estimation error and provide visual uncertainty heat maps to guide deployment decisions. To highlight these additions, we provide a brief description of the mathematical algorithms and theory underlying these new capabilities. We demonstrate the usage of new features with illustrative code examples and include practical advice for implementation across various application domains. Finally, we outline a roadmap of potential extensions to further enhance the package's functionality and applicability to emerging sensing challenges.

# Statement of need
Sensor placement is critical for efficient monitoring, control, and decision-making in modern engineering systems. Sensors play a crucial role in characterizing spatio-temporal dynamics in high-dimensional, non-linear systems such as fluid flows [@erichson2020shallow], manufacturing [@manohar2018predicting], geophysical [@alonso2010novel] and nuclear systems [@karnik2024constrained]. Optimal sensor placement ensures accurate, real-time tracking of key system variables with minimal hardware and enables cost-effective, real-time system analysis and control. In general, sensor placement optimization is NP-hard and cannot be solved in polynomial time.  There are ${n \choose p} = n!/((n-p)!p!)$ possible combinations of choosing $p$ sensors from an $n$-dimensional state.
Common approaches to optimizing sensor placement include maximizing the information criteria [@krause2008near], Bayesian Optimal Experimental Design [@alexanderian2021optimal], compressed sensing [@donoho2006compressed], and heuristic methods. Many sensor placement methods have submodular objective form, which sets guarantees on how close an efficient greedy placement can be to the unknown true optimum [@summers2015submodularity]. Sub-modular objectives can be efficiently optimized for hundreds or thousands of candidate locations using convex [joshi2008sensor] or greedy optimization approaches [@summers2015submodularity] .

![An overview image of capabilities of Pysensors](../Fig1.jpeg "PySensors 2.0 expands its capabilities by introducing custom basis functions, optimizers, constraints, solvers, and uncertainty quantification, enabling constrained sensing, over- and under-sampling,  and uncertainty quantification in the presence of noisy sensor measurements.")


`PySensors` is a Python package [@de2021pysensors] dedicated to solving the complex challenge of optimal sensor placement in data-driven systems. It implements advanced sparse optimization algorithms that use dimensionality reduction techniques to identify the most informative measurement locations with remarkable efficiency [@manohar2018data;@brunton2016sparse;@clark2020multi]. It helps users identify the best locations for sensors when working with complex high dimensional data, focusing on both reconstruction [@manohar2018data] and classification [@brunton2016sparse] tasks. The package follows `scikit-learn` conventions for user-friendly access while offering advanced customization options for experienced users. Designed with researchers and practitioners in mind, `PySensors` provides open-source, accessible tools that support model discovery across various scientific applications.

This new version of `Pysensors` focuses specifically on practical engineering applications where measurement data is inherently noisy and spatial deployment constraints are unavoidable. Key improvements include constraint-aware optimization methods that handle spatial restrictions and sensor density limitations. In addition, the framework introduces uncertainty quantification metrics that track how measurement noise propagates through reconstruction algorithms, enabling robust error estimation in sensor outputs. This version of `Pysensors` implements methodologies introduced by Klishin et al. [@klishin2023data] and Karnik et al. [@karnik2024constrained] to make them accessible to scientists and engineers in all domains. These enhancements transform `PySensors` from a purely academic tool into a practical platform for solving real-world sensing challenges while maintaining mathematical rigor.

Pysensors implements `Sparse Sensor Placement Optimization for Reconstruction (SSPOR)`[@manohar2018data] of full fields $\mathbf{x}\in\mathbb{R}^n$ from $p$ noise-corrupted sensor measurements $\mathbf{y}\in\mathbb{R}^p$
$$
\mathbf{y} =  \mathbb{S} \mathbf{x} + \boldsymbol\eta
$$
where $\boldsymbol{\eta}$ consists of zero-mean, Gaussian independent and identically distributed (i.i.d.) components, and $\mathbb{S} \in \mathbb{R}^{p\times n}$ is the desired sensor (measurement) selection operator.
This measurement selection operator $\mathbb{S}$ encodes point measurements with unit entries in a sparse matrix $\mathbb{S} = \begin{bmatrix} \mathbf{e}_{\gamma_1} & \mathbf{e}_{\gamma_2} & \hdots & \mathbf{e}_{\gamma_ p}\end{bmatrix}^T$ where $\mathbf{e}_j$ are canonical basis vectors for $\mathbb{R}^n$, with a unit entry in component $j$ (where a sensor should be placed) and zeroes elsewhere. Here, $\gamma = \{\gamma_1, \gamma_2, \hdots, \gamma_p\} \subset \{1, 2, \hdots, n\}$ denotes the index set of sensor locations with cardinality $p$.
Sensor selection then corresponds to the components of $\mathbf{x}$ that were chosen to be measured: $\mathbf{\mathbb{S} x} = \begin{bmatrix} x_{\gamma_1} & x_{\gamma_2} & \hdots & x_{\gamma_p}\end{bmatrix}^T$.
The `SSPOR` class selects these sensors through a cohesive computational framework for strategically minimizing sensor deployment while maintaining reconstruction fidelity in high-dimensional systems. When initialized with measurement data $\mathbf{x}\in\mathbb{R}^n$, `SSPOR` first applies dimensional reduction through basis identification, then employs the computationally efficient QR factorization algorithm to determine optimal sensor locations. This approach, which has demonstrated efficacy in both reduced-order modeling and sparse sensing applications, strategically leverages the inherent structure of the identified basis to prioritize the most informative measurement points.

The `Sparse Sensor Placement Optimization for Classification (SSPOC)` framework identifies minimal sensor configurations that can classify high dimensional signals $\mathbf{x}\in\mathbb{R}^n$ as one of $c$ classes. Unlike traditional compressed sensing [@baraniuk2010model;@donoho2006compressed;@candes2006stable] approaches that focus on signal reconstruction, `SSPOC` specifically targets the preservation of classification boundaries in feature space $\boldsymbol{\Psi}_r$ by identifying the sparsest sensor set capable of reconstructing discriminant hyperplanes.
The `SSPOC` architecture accepts any combination of basis representation and linear classifier, defaulting to Linear Discriminant Analysis (LDA) and Identity basis when not otherwise specified. The optimization pipeline proceeds through multiple stages: dimensional reduction via basis fitting, classifier training within the reduced space, sparse optimization incorporating both classifier weights and basis structures, sensor selection based on optimization outputs, and optional classifier retraining using only measurements from the selected locations.

High dimensional field dynamics $\mathbf{x}\in\mathbb{R}^n$ can be represented as a linear combination of spatial basis modes $\boldsymbol{\Psi}$ weighted by time-varying coefficients $\mathbf{a}$
$$\mathbf{x} = \boldsymbol{\Psi}_r\mathbf{a}.$$
This basis, which can be built from spectral or data-driven decomposition methods, is typically chosen so that the embedding dimension is as small as possible, i.e., $r\ll n$. Different basis functions can significantly impact sensor selection effectiveness and reconstruction quality [@manohar2018data]. PySensors offers several interchangeable `basis` options for sparse sensor selection:
- `Identity`: Processes raw measurement data directly without transformation.
- `SVD`: Utilizes truncated singular value decomposition's $\mathbf{X} = \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}^*_r$ left singular vectors $\mathbf{U}_r$, computing only the specified number of modes to reduce computational demands. A randomized SVD option further enhances efficiency.
- `RandomProjection`: Projects measurements into a new space by multiplying them with random Gaussian vectors, connecting to compressed sensing methodologies from established literature [@baraniuk2010model;@donoho2006compressed;@candes2006stable].
- `Custom`: This is a new option included in `Pysensors 2.0` that enables users to leverage custom basis functions beyond `PySensors` built-in options. Researchers can transform their data into an alternative basis such as dynamic mode decomposition modes [@schmid2010dynamic], before passing it to a PySensors instance configured with the Custom basis.

When a user specifies $p$ sensors, SSPOR first fits a basis $\boldsymbol{\Psi}_r$ to the data and optimizes sensor placement by minimizing the following objective function:
$$
\gamma_* = \argmax_{\gamma, |\gamma|= p} \; \log \det((\mathbb{S} \boldsymbol{\Psi}_r)^T(\mathbb{S} \boldsymbol{\Psi}_r)).\label{eq:detobj} \tag{1}
$$
where $\gamma_*$ denotes the index set of optimized sensor locations with cardinality $p$. When $p = r$, is equivalent to the maximizer of $\log |\det(\mathbb{S} \boldsymbol{\Psi}_r)|$.
Direct optimization of this criterion leads to a brute force combinatorial search. This sensor placement approach builds upon the empirical interpolation method (EIM) [@barrault2004empirical] and discrete empirical interpolation method (DEIM) [@chaturantabut2010nonlinear] to develop a greedy strategy for optimizing sensor selection built upon the pivoted QR factorization[@drmac2016new;@manohar2018data;@manohar2018predicting;@manohar2021optimal].

PySensors implements the `QR` optimizer for optimal sensor selection in unconstrained scenarios where the number of sensors $p$ equals the number of modes $r$. The framework incorporates heterogeneous cost functions into the optimization process to accommodate practical deployment constraints. For example, when monitoring oceanographic parameters, the system can account for the substantially higher costs of deep-sea sensors relative to coastal installations[@clark2020multi]. This capability is implemented through the `Cost-Constrained QR (CCQR)` algorithm in the optimizers submodule, allowing users to balance information capture against resource limitations when designing sensor networks for complex physical systems.

Traditional QR factorization presents challenges in under-sampling $p < r$ and over-sampling $p > r$ scenarios, as well as when spatial constraints must be considered. `PySensors 2.0` addresses these limitations through two new optimization algorithms: `Generalized QR (GQR)` and `Two Point GReedy (TPGR)`.

# New Features

`PySensors` has been enhanced to address critical challenges in sensor placement by incorporating spatial constraints, noise-induced uncertainties, over/under sampling and sensor interactions. These advancements optimize reconstruction performance for complex processes across nuclear energy, fluid dynamics, biological systems, and manufacturing applications, enabling more accurate modeling, prediction, and control capabilities.
The previous version of `PySensors` implements two sensor placement approaches: (1) an unconstrained optimization formulated as `QR` that allows sensors to be placed anywhere in the domain, and (2) a cost-constrained optimization formulated as `CCQR` that incorporates variable placement costs, making certain regions more expensive for sensor deployment.

Many engineering applications have extreme operating conditions, high costs, limited accessibility and safety regulations that impose significant constraints on spatial locations of sensors. Implementing spatially constrained sensor placement requires a deeper intervention in the underlying QR optimization framework. To address this requirement, we have developed a new optimization functionality called General QR `GQR` based on Karnik et. al. [@karnik2024constrained], which provides the architectural flexibility needed to handle complex spatial constraints. In `Pysensors 2.0` we enhance the algorithm's capabilities by incorporating diverse spatial constraints defined by users through `_norm_calc.py` in `utils`. The three types of spatial constraints handled by the algorithm are:

- *Region constrained:* This type of constraint arises when we can place either a _maximum_ of or _exactly_ $s$ sensors in a certain region, while the remaining $r-s$ sensors must be placed outside the constraint region.
  - *Maximum:* This case deals with applications in which the number of sensors in the constraint region should be less than or equal to $s$. This functionality has been implemented through `max_n` in `_norm_calc.py`.
  - *Exact:* This case deals with applications in which the number of sensors in the constraint region should equal $s$. This functionality has been implemented through `exact_n` in `_norm_calc.py`.
- *Predetermined:* This type of constraint occurs when a certain number of sensor locations $s$ are already specified, and  optimized locations for the remaining sensors are desired. This functionality has been implemented through `predetermined` in `_norm_calc.py`.
- *Distance constrained:* This constraint enforces a minimum distance $d$ between selected sensors. This functionality has been implemented through `distance` in `_norm_calc.py`.

To implement spatial constraints in our sensor placement methodology, we must first define the shape of our constraint region and then identify which grid points fall within the designated constraint regions. We accomplish this using the specialized classes and functions provided in the `_constraints.py` module in `utils`. The classes define constraint regions in various geometric shapes including `Circle`, `Ellipse`, `Polygon`, `Parabola`, `Line`, and `Cylinder`. Additionally, we provide `UserDefinedConstraints` options that allow users to supply their own constraints either as a Python`.py` file containing their constraint shape definition or as an equation string. These classes work with two distinct data formats: image data defined by pixel coordinates and tabular data stored in dataframes with explicit x, y, and z values. Upon instantiation of the class, users must provide the input `data` in one of two formats: a matrix representation for images or a structured dataframe for tabular data. In the latter case, the class requires additional keyword arguments `kwargs` specifying the columns corresponding to `X_axis`, `Y_axis`, `Z_axis`, and `Field` parameters, thereby enabling the extraction of relevant data from their respective columns for subsequent analysis.

The D-optimal objective in \ref{detobj} suffers from two limitations: it is not defined for the under-sampling case $p<r$, and it is hard to interpret and visualize directly. Ref.[@klishin2025doubledescent] resolves these limitations by adding a prior regularization and decomposing the resulting objective into sums over the placed sensors:
$$
\mathcal{H}\equiv -\log \det(\mathbf{S}^{-2}+(\mathbb{S} \boldsymbol{\Psi}_r)^T(\mathbb{S} \\boldsymbol{\Psi}_r)/\eta^2)\approx E_b+\sum\limits_{i\in \gamma}h_i+\sum\limits_{i\neq j \in\gamma}J_{ij} \label{eq:TPGR} \tag{2}
$$
where $\mathbf{S}$ is the assumed prior covariance matrix of the coefficients $\mathbf{a}$ and $\eta$ is the assumed sensor noise magnitude. The typical prior covariances are $\mathbf{S}\propto \mathbf{I}$ (isotropic Gaussian) or $\mathbf{S}=\mathbf{\Sigma_r}/\sqrt{N}$ (normalized singular values of training data). The series expansion over terms with more sensors is in principle exact, but we approximate it with the first two terms. In the approximation, $E_b$ is a constant term that does not affect sensor selection, $h_i,J_{ij}$ are the interaction landscapes computed from the basis and the prior. The objective in \ref{eq:TPGR} involves only summation over the selected sensors and is thus cheaper to evaluate and update than the original objective in Eqn. \ref{eq:detobj}. The approximate objective is used in the Two Point GReedy (TPGR) optimizer that can return a user-specified number of sensors $p$ for any mode number $r$. In contrast, the QR algorithm returns exactly $p=r$ sensors in order of decreasing importance through pivoting, and all following sensors are random. The sensor sets returned by `TPGR` are nearly equivalent to QR for isotropic prior $\mathbf{S}\propto \mathbf{I}$ and small noise $\eta$.

Once the set of $p$ sensors has been determined using any of the methods, the sensor measurements $\mathbf{y}$ can be used to determine the state coefficients $\mathbf{a}$. Under the assumption of linearity, the reconstruction always takes the shape $\hat{\mathbf{a}}=\mathbf{A}\mathbf{y}$ for some matrix $\mathbf{A}:r\times p$.
The first version of the reconstruction matrix corresponds to the Least Squares solution via the Moore-Penrose pseudoinverse:
$$
    \mathbf{A}_{LS}=(\mathbb{S} \boldsymbol{\Psi}_r)^\dagger.
$$

`PySensors 2.0` adds the Regularized Least Squares solution derived in Ref.~\cite{klishin2023data}:
$$
    \mathbf{A}_{RLS}=\left( \mathbf{S}^{-2}+(\mathbb{S} \boldsymbol{\Psi}_r)^T(\mathbb{S} \boldsymbol{\Psi}_r)/\eta^2 \right)^{-1} (\mathbb{S} \boldsymbol{\Psi}_r)^T/\eta^2,
$$
where similarly to the TPGR optimizer, $\mathbf{S}$ is the assumed prior covariance matrix and $\eta$ is the assumed sensor noise magnitude. The relative magnitude of the coefficient prior and the noise determines whether the reconstruction primarily relies on the measurements or the prior information. This Regularized Least Squares solution is now the default reconstruction solver for PySensors 2.0. The 'unregularized' method will use the Least Squares method using Moore-Penrose pseudoinverse reconstruction solver.

For any choice of the reconstruction matrix $\mathbf{A}$, the reconstruction $\hat{\mathbf{a}}=\mathbf{A}\mathbf{y}$ predicts the \emph{most likely} coefficients of the reconstructed state, from which the full state can be obtained via projection $\hat{\mathbf{x}}=\boldsymbol{\Psi}_r\hat{\mathbf{a}}$. However, the reconstruction is sensitive to the sensor noise. The expected error of the reconstruction is captured by the covariance matrix derived in Ref. [@klishin2023data]:
\begin{align}
    \mathbf{K}=\boldsymbol{\Psi}_r \mathbf{B} \mathbf{B}^T \boldsymbol{\Psi}_r^T;\quad \mathbf{B}=\eta  \mathbf{A}. \label{eqn:K}
\end{align}

![A flowchart to suggest which method to use](../Fig2.jpeg "When selecting a sensing method in PySensors, consider your primary objective: For field reconstruction in standard settings, use QR with Identity or SVD basis. For classification tasks, leverage SVD basis with SSPOC optimizer. When facing spatial constraints, choose GQR optimizer. For under-sampling (p < r) and over-sampling cases (p > r)scenarios , select TPGR optimizer. In noisy environments enable uncertainty quantification for robust results.")


# Acknowledgments
The authors acknowledge support from the Boeing Company, NSF AI Institute in Dynamic Systems under grant 2112085 and through the Idaho National Laboratory (INL) Laboratory Directed Research \& Development (LDRD) Program under DOE Idaho Operations Office Contract DE-AC07-05ID14517 for LDRD-22A1059-091FP.

# References