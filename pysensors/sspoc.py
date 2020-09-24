"""
SSPOC object definition.
"""
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
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

    def __init__(
        self,
        basis=None,
        classifier=None,
        n_sensors=None,
        threshold=None,
        l1_penalty=0.1,
    ):
        if basis is None:
            basis = Identity()
        self.basis = basis
        if classifier is None:
            classifier = LinearDiscriminantAnalysis()
        self.classifier = classifier
        self.n_sensors = n_sensors
        self.threshold = threshold
        self.l1_penalty = l1_penalty
        self.n_basis_modes = None

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
        with warnings.catch_warnings():
            action = "ignore" if quiet else "default"
            warnings.filterwarnings(action, category=UserWarning)
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

        if self.threshold is None:
            # Chosen as in Brunton et al. (2016)
            threshold = np.sqrt(np.sum(s ** 2)) / (
                2 * self.basis_matrix_inverse_.shape[0] * n_classes
            )
        else:
            threshold = self.threshold

        # Decide which sensors to retain based on s
        self.sensor_coef_ = s
        self.sparse_sensors_ = np.array([])
        xy = (x, y) if refit else None
        self.update_sensors(n_sensors=self.n_sensors, threshold=threshold, xy=xy)

        # Form a dummy classifier for when no sensors are retained
        self.dummy_ = DummyClassifier(strategy="stratified")
        self.dummy_.fit(x[:, 0], y)

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
        if self.n_sensors == 0:
            warnings.warn(
                "SSPOC model has no selected sensors so predictions are random. "
                "Increase n_sensors or lower threshold with SSPOC.update_sensors."
            )
            return self.dummy_.predict(x[:, 0])
        if self.refit_:
            return self.classifier.predict(x)
        else:
            return self.classifier.predict(np.dot(x, self.basis_matrix_inverse_.T))

    def update_sensors(
        self, n_sensors=None, threshold=None, xy=None, method=np.max, **method_kws
    ):
        check_is_fitted(self, "sensor_coef_")
        if n_sensors is not None and threshold is not None:
            warnings.warn(
                f"Both n_sensors({n_sensors}) and threshold({threshold}) were passed "
                "so threshold will be ignored"
            )

        if n_sensors is None and threshold is None:
            raise ValueError("At least one of n_sensors or threshold must be passed.")

        elif n_sensors is not None:
            if n_sensors > len(self.sensor_coef_):
                raise ValueError(
                    f"n_sensors({n_sensors}) cannot exceed number of available sensors "
                    f"({len(self.sensor_coef_)})"
                )
            self.n_sensors = n_sensors
            # Could be made more efficient with a max heap
            if np.ndim(self.sensor_coef_) == 1:
                sorted_sensors = np.argsort(-np.abs(self.sensor_coef_))
                if np.abs(self.sensor_coef_[sorted_sensors[-1]]) == 0:
                    warnings.warn(
                        "Some uninformative sensors were selected. "
                        "Consider decreasing n_sensors"
                    )
            else:
                sorted_sensors = np.argsort(
                    -method(np.abs(self.sensor_coef_), axis=1, **method_kws)
                )
                if (
                    method(
                        np.abs(self.sensor_coef_[sorted_sensors[-1], :]), **method_kws
                    )
                    == 0
                ):
                    warnings.warn(
                        "Some uninformative sensors were selected. "
                        "Consider decreasing n_sensors"
                    )
            self.sparse_sensors_ = sorted_sensors[:n_sensors]

        else:
            self.threshold = threshold
            if np.ndim(self.sensor_coef_) == 1:
                sparse_sensors = np.nonzero(np.abs(self.sensor_coef_) >= threshold)[0]
            else:
                sparse_sensors = np.nonzero(
                    method(np.abs(self.sensor_coef_), axis=1, **method_kws) >= threshold
                )[0]

            self.n_sensors = len(sparse_sensors)
            self.sparse_sensors_ = sparse_sensors

            if self.n_sensors == 0:
                warnings.warn(
                    f"Threshold set too high ({threshold}); no sensors selected."
                )

        # Refit if xy was passed
        if xy is not None:
            if self.n_sensors > 0:
                x, y = xy
                self.classifier.fit(x[:, self.sparse_sensors_], y)
                self.refit_ = True
            else:
                warnings.warn("No selected sensors; model was not refit.")

    @property
    def selected_sensors(self):
        check_is_fitted(self, "sparse_sensors_")
        return self.sparse_sensors_
