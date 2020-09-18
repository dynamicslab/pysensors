"""
SSPOC object definition.
"""
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import check_is_fitted

from .basis import Identity
from .utils import constrained_binary_solve
from .utils import constrained_multiclass_solve
from .utils import validate_input


INT_TYPES = (int, np.int64, np.int32, np.int16, np.int8)


class SSPOC(BaseEstimator):
    """
    Sparse Sensor Placement Optimization for Classification.

    Parameters
    ----------
    TODO
    """

    def __init__(self, basis=None, classifier=None, threshold=None, l1_penalty=1.0):
        if basis is None:
            basis = Identity()
        self.basis = basis
        if classifier is None:
            classifier = LinearDiscriminantAnalysis()
        self.classifier = classifier
        self.n_basis_modes = None
        self.threshold = threshold
        self.l1_penalty = l1_penalty

    def fit(
        self,
        x,
        y,
        quiet=False,
        prefit_basis=False,
        seed=None,
        refit=True,
        **optimizer_kws,
    ):
        """
        Fit the SSPOC model, determining which sensors are relevant.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_input_features)
            Training data.

        y: array-like, shape (n_samples,)
            Training labels.

        quiet: boolean, optional (default False)
            Whether or not to suppress warnings during fitting.

        prefit_basis: boolean, optional (default False)
            Whether or not the basis has already been fit to x.
            For example, you may have already fit and experimented with
            a ``POD`` object to determine the optimal number of modes. This
            option allows you to avoid an unnecessary SVD.

        seed: int, optional (default None)
            Seed for the random number generator used to shuffle sensors after the
            ``self.basis.n_basis_modes`` sensor. Most optimizers only rank the top
            ``self.basis.n_basis_modes`` sensors, leaving the rest virtually
            untouched. As a result the remaining samples are randomly permuted.

        refit: bool, optional (default True)
            Whether or not to refit the classifier using measurements
            only from the learned sensor locations.

        optimizer_kws: dict, optional
            Keyword arguments to be passed to the optimization routine.
        """

        # Fit basis functions to data
        # TODO: base class should have a _fit_basis method
        if prefit_basis:
            check_is_fitted(self.basis, "basis_matrix_")
        else:
            x = validate_input(x)

            with warnings.catch_warnings():
                action = "ignore" if quiet else "default"
                warnings.filterwarnings(action, category=UserWarning)
                self.basis.fit(x)

        # Get matrix representation of basis - this is \Psi^T in the paper
        # TODO: implement this method
        self.basis_matrix_inverse_ = self.basis.matrix_inverse(
            n_basis_modes=self.n_basis_modes
        )

        # Find weight vector
        # Equivalent to np.dot(self.basis_matrix_inverse_, x.T).T
        # TODO
        self.classifier.fit(np.matmul(x, self.basis_matrix_inverse_.T), y)
        # self.classifier.fit(np.dot(self.basis_matrix_.T, x), y)
        # self.optimizer.fit(self.basis_matrix_.T, y)

        w = np.squeeze(self.classifier.coef_).T

        n_classes = len(set(y[:]))
        if n_classes == 2:
            s = constrained_binary_solve(w, self.basis_matrix_inverse_, **optimizer_kws)
        else:
            s = constrained_multiclass_solve(
                w, self.basis_matrix_inverse_, alpha=self.l1_penalty, **optimizer_kws
            )

        # Get sensor locations from s
        if self.threshold is None:
            threshold = 1  # TODO - pick this as in the paper
        else:
            threshold = self.threshold

        # Decide which sensors to retain
        self.sensor_coef_ = s
        self.update_threshold(threshold)

        # Refit the classifier using sparse measurements
        if refit:
            self.classifier.fit(x[:, self.sparse_sensors_], y)
            self.refit_ = True

        return self

    def predict(self, x):
        """
        Predict classes for given measurements.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_sensors) or (n_samples, n_features)
            Examples to be classified.
            The measurements should be taken at the sensor locations specified by
            ``self.selected_sensors``.

        Returns
        -------
        y: numpy array, shape (n_samples,)
            Predicted classes.
        """
        check_is_fitted(self, "sensor_coef_")
        if self.refit_:
            return self.classifier.predict(x)
        else:
            return self.classifier.predict(np.dot(x, self.basis_matrix_inverse_.T))

    def update_threshold(self, threshold, xy=None, method=np.mean, **method_kws):
        check_is_fitted(self, "sensor_coef_")
        self.threshold = threshold
        if np.ndim(self.sensor_coef_) == 1:
            sparse_sensors = np.nonzero(np.abs(self.sensor_coef_) > threshold)[0]
        else:
            # We just need to consider the first column of self.sensor_coef_
            # since MultiTaskLasso (group lasso) zeros out entire rows

            sparse_sensors = np.nonzero(
                method(np.abs(self.sensor_coef_), axis=1, **method_kws) > threshold
            )[0]

        # Don't save new sensors unless
        if np.count_nonzero(sparse_sensors) == 0:
            warnings.warn(f"Threshold set too high ({threshold}); no sensors selected.")
            if xy is not None:
                warnings.warn("No selected sensors; model was not refit.")
        else:
            self.sparse_sensors_ = sparse_sensors

        # Refit if xy was passed
        if xy is not None:
            x, y = xy
            self.classifier.fit(x[:, self.sparse_sensors_], y)
            self.refit_ = True

    @property
    def selected_sensors(self):
        check_is_fitted(self, "sparse_sensors_")
        return self.sparse_sensors_