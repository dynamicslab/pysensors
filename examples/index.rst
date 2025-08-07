PySensors Examples
==================

This directory provides examples of how to use `PySensors` objects to solve sensor placement problems.
`PySensors` was designed to be completely compatible with `scikit-learn`.

`PySensors Overview <./pysensors_overview.ipynb>`__
---------------------------------------------------
This notebook gives an overview of most of the different tools available in `PySensors`. It's a good place to start to get a quick idea of what the package is capable of.

.. toctree::
  :hidden:
  :maxdepth: 1

  pysensors_overview

`Classification <./classification.ipynb>`__
-------------------------------------------
This notebook showcases the use of `SSPOC` class (Sparse Sensor Placement Optimization for Classification) to choose sparse sets of sensors for *classification* problems.

.. toctree::
  :hidden:
  :maxdepth: 1

  classification

Reconstruction
--------------
These notebooks show how the `SSPOR` class (Sparse Sensor Placement Optimization for Reconstruction) can be used with different optimizers.
The default optimizer for `SSPOR` is `QR`, which uses QR pivoting to select sensors in unconstrained problems.

`GQR` (General QR) optimizer provides a more intrusive approach into the `QR` pivoting procedure to take into account spatial constraints. The `General QR Optimizer for Spatial Constraints <./spatial_constrained_qr.ipynb>`__ and `Functional Constraints for Olivetti Faces <./Olivetti_constrained_sensing.ipynb>`__ notebooks provide a detailed account of unconstrained and constrained sensor placement.

`CCQR` (Cost Constrained QR) optimizer can be used to place sparse sensors when there are variable costs associated with different locations. The `Cost Constrained QR <./cost_constrained_qr.ipynb>`__ notebook showcases the `CCQR` optimizer.

`TPGR` (Two Point GReedy) optimizer uses a thermodynamic approach to sensor placement that maps the complete landscape of sensor interactions induced by the training data and places sensors such that the marginal energy of each next placed sensor is minimized. The `TPGR <./two_point_greedy.ipynb>`__ notebook goes into detail about the optimizer and the one-point and two-point enery landscape computation. The `TPGR` optimizer requires prior and noise.

There are two methods used for reconstruction: `Unregularized Reconstruction`, which uses the Moore-Penrose Pseudoinverse method, and `Regularized Reconstruction`, that uses a maximal likelihood reconstructor that requires a prior and noise.
The `Reconstruction Comparison <./reconstruction_comparison.ipynb>`__ notebook compares these two methods using the `TPGR` optimizer. It also shows a comparison between `TPGR` and `QR` optimizers using both of the reconstruction methods.

.. toctree::
  :hidden:
  :maxdepth: 1

  spatial_constrained_qr
  Olivetti_constrained_sensing
  cost_constrained_qr
  two_point_greedy
  reconstruction_comparison

Basis
-----
The `Basis Comparison <./basis_comparison.ipynb>`__ notebook compares the different basis options implemented in `PySensors` on a simple problem.
`Cross Validation <./cross_validation.ipynb>`__ is also performed with `scikit-learn` objects to optimize the number of sensors and/or basis modes.

.. toctree::
  :hidden:
  :maxdepth: 1

  basis_comparison
  cross_validation

Applications
------------
These notebooks showcase the sensor placement optimization methods on datasets ranging from `Sea Surface Temperature <./sea_surface_temperature.ipynb>`__ to predicting the temperature within a `Fuel Rod <./OPTI-TWIST_constrained_sensing.ipynb>`__ with spatially constrained sensors.
The `Polynomial Curve Fitting <./polynomial_curve_fitting.ipynb>`__ notebook demonstrates how to use PySensors to select sensor locations for polynomial interpolation using the monomial basis :math:`1, x, x^2, x^3, \dots, x^k`.

.. toctree::
  :hidden:
  :maxdepth: 1

  sea_surface_temperature
  polynomial_curve_fitting
  OPTI-TWIST_constrained_sensing
