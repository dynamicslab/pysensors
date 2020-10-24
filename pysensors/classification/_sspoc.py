"""
Sparse Sensor Placement Optimization for Classification (SSPOC) based
on

    Brunton, Bingni W., et al.
    "Sparse sensor placement optimization for classification."
    SIAM Journal on Applied Mathematics 76.5 (2016): 2099-2122.

See also the following paper for improvements on this method

    Mohren, Thomas L., et al.
    "Neural-inspired sensors enable sparse, efficient classification
    of spatiotemporal data."
    Proceedings of the National Academy of Sciences
    115.42 (2018): 10564-10569.
"""
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_is_fitted

from ..basis import Identity
from ..utils import constrained_binary_solve
from ..utils import constrained_multiclass_solve
from ..utils import validate_input


INT_DTYPES = (int, np.int64, np.int32, np.int16, np.int8)


class SSPOC(BaseEstimator):
    r"""
    Sparse Sensor Placement Optimization for Classification (SSPOC) object.

    As the name suggests, this class can be used to select optimal sensor
    locations (measurement locations) for classification tasks.

    See the following reference for more information:

        Brunton, Bingni W., et al.
        "Sparse sensor placement optimization for classification."
        SIAM Journal on Applied Mathematics 76.5 (2016): 2099-2122.

    Parameters
    ----------
    basis: basis object, optional (default :class:`pysensors.basis.Identity`)
        Basis in which to represent the data. Default is the identity basis
        (i.e. raw features).

    classifier: classifier object, optional \
            (default Linear Discriminant Analysis (LDA))
        Classifier for which to optimize sensors. Must be a *linear* classifier
        with a :code:`coef_` attribute and :code:`fit` and :code:`predict`
        methods.

    n_sensors: positive integer, optional (default None)
        Number of sensor locations to be used after fitting.
        If :code:`n_sensors` is not None then it overrides the :code:`threshold`
        parameter.
        If set to 0, then :code:`classifier` will be replaced with a dummy
        classifier which predicts the class randomly.

    threshold: nonnegative float, optional (default None)
        Threshold for selecting sensors.
        Overriden by :code:`n_sensors`.
        If both :code:`threshold` and :code:`n_sensors` are None when the
        :meth:`fit` method is called, then the threshold will be set to

        .. math::
            \frac{\|s\|_F}{2rc}

        where :math:`s` is a sensor coefficient matrix, :math:`r` is the number
        of basis modes, and :math:`c` is the number of distinct classes,
        as suggested in Brunton et al. (2016).

    l1_penalty: nonnegative float, optional (default 0.1)
        The L1 penalty term used to form the sensor coefficient matrix, s.
        Larger values will result in a sparser s and fewer selected sensors.
        This parameter is ignored for binary classification problems.

    Attributes
    ----------
    n_basis_modes: nonnegative integer
        Number of basis modes to be used when deciding sensor locations.

    basis_matrix_inverse_: np.ndarray, shape (n_basis_modes, n_input_features)
        The inverse of the matrix of basis vectors.

    sensor_coef_: np.ndarray, shape (n_input_features, n_classes)
        The sensor coefficient matrix, s.

    sparse_sensors_: np.ndarray, shape (n_sensors, )
        The selected sensors.

    Examples
    --------
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.datasets import make_classification
    >>> from pysensors.classification import SSPOC
    >>>
    >>> x, y = make_classification(n_classes=3, n_informative=3, random_state=10)
    >>>
    >>> model = SSPOC(n_sensors=10, l1_penalty=0.03)
    >>> model.fit(x, y, quiet=True)
    SSPOC(basis=Identity(n_basis_modes=100),
          classifier=LinearDiscriminantAnalysis(), l1_penalty=0.03, n_sensors=10)
    >>> print(model.selected_sensors)
    [10 13  6 19 17 16 15 14 12 11]
    >>>
    >>> acc = accuracy_score(y, model.predict(x[:, model.selected_sensors]))
    >>> print("Accuracy:", acc)
    Accuracy: 0.66
    >>>
    >>> model.update_sensors(n_sensors=5, xy=(x, y), quiet=True)
    >>> print(model.selected_sensors)
    [10 13  6 19 17]
    >>>
    >>> acc = accuracy_score(y, model.predict(x[:, model.selected_sensors]))
    >>> print("Accuracy:", acc)
    Accuracy: 0.6
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
            a ``SVD`` object to determine the optimal number of modes. This
            option allows you to avoid an unnecessary SVD.

        refit: boolean, optional (default True)
            Whether or not to refit the classifier using measurements
            only from the learned sensor locations.

        optimizer_kws: dict, optional
            Keyword arguments to be passed to the optimization routine.

        Returns
        -------
        self: a fitted :class:`SSPOC` instance
        """

        # Fit basis functions to data
        if prefit_basis:
            check_is_fitted(self.basis, "basis_matrix_")
        else:
            x = validate_input(x)

            with warnings.catch_warnings():
                action = "ignore" if quiet else "default"
                warnings.filterwarnings(action, category=UserWarning)
                self.basis.fit(x)

        # Get matrix representation of basis - this is \Psi^T in the paper
        self.basis_matrix_inverse_ = self.basis.matrix_inverse(
            n_basis_modes=self.n_basis_modes
        )

        # Find weight vector
        with warnings.catch_warnings():
            action = "ignore" if quiet else "default"
            warnings.filterwarnings(action, category=UserWarning)
            self.classifier.fit(np.matmul(x, self.basis_matrix_inverse_.T), y)

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
        self.update_sensors(
            n_sensors=self.n_sensors, threshold=threshold, xy=xy, quiet=quiet
        )

        # Form a dummy classifier for when no sensors are retained
        self.dummy_ = DummyClassifier(strategy="stratified")
        self.dummy_.fit(x[:, 0], y)

        return self

    def predict(self, x):
        """
        Predict classes for given measurements.
        If :code:`self.n_sensors` is 0 then a dummy classifier is used in place
        of :code:`self.classifier`.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_sensors) or (n_samples, n_features)
            Examples to be classified.
            The measurements should be taken at the sensor locations specified by
            ``self.selected_sensors``.

        Returns
        -------
        y: np.ndarray, shape (n_samples,)
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
        self,
        n_sensors=None,
        threshold=None,
        xy=None,
        quiet=False,
        method=np.max,
        **method_kws,
    ):
        """
        Update the selected sensors by changing either the preferred number of sensors
        or the threshold used to select the sensors, refitting the classifier
        afterwards, if possible.

        Parameters
        ----------
        n_sensors: nonnegative integer, optional (default None)
            The number of sensor locations to select.
            If None, then :code:`threshold` will be used to pick the sensors.
            Note that :code:`n_sensors` and :code:`threshold` cannot both be None.

        threshold: nonnegative float, optional (default None)
            The threshold to use to select sensors based on the magnitudes of entries
            in :code:`self.sensor_coef_` (s).
            Overridden by :code:`n_sensors`.
            Note that :code:`n_sensors` and :code:`threshold` cannot both be None.

        xy: tuple of np.ndarray, length 2, optional (default None)
            Tuple containing training data x and labels y for refitting.
            x should have shape (n_samples, n_input_features) and y shape (n_samples, ).
            If not None, the classifier will be refit after the new sensors have been
            selected.

        quiet: boolean, optional (default False)
            Whether to silence warnings.

        method: callable, optional (default :code:`np.max`)
            Function used along with :code:`threshold` to select sensors.
            For binary classification problems one need not specify a method.
            For multiclass classification problems, :code:`sensor_coef_` (s) has
            multiple columns and :code:`method` is applied along each row to aggregate
            coefficients for thresholding, i.e. :code:`method` is called as follows
            :code:`method(np.abs(self.sensor_coef_), axis=1, **method_kws)`.
            Other examples of acceptable methods are :code:`np.min`, :code:`np.mean`,
            and :code:`np.median`.

        **method_kws: dict, optional
            Keyword arguments to be passed into :code:`method` when it is called.
        """
        check_is_fitted(self, "sensor_coef_")
        warn = not quiet

        if n_sensors is not None and threshold is not None and warn:
            warnings.warn(
                f"Both n_sensors({n_sensors}) and threshold({threshold}) "
                "were passed so threshold will be ignored"
            )

        if n_sensors is None and threshold is None:
            raise ValueError("At least one of n_sensors or threshold must be passed.")

        elif n_sensors is not None:
            if n_sensors > len(self.sensor_coef_):
                raise ValueError(
                    f"n_sensors({n_sensors}) cannot exceed number of available "
                    f"sensors ({len(self.sensor_coef_)})"
                )
            self.n_sensors = n_sensors
            # Could be made more efficient with a max heap
            # (we don't need to sort the whole list)
            if np.ndim(self.sensor_coef_) == 1:
                sorted_sensors = np.argsort(-np.abs(self.sensor_coef_))
                if (
                    np.abs(self.sensor_coef_[sorted_sensors[n_sensors - 1]]) == 0
                    and warn
                ):
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
                        np.abs(self.sensor_coef_[sorted_sensors[n_sensors - 1], :]),
                        **method_kws,
                    )
                    == 0
                    and warn
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

            if self.n_sensors == 0 and warn:
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

    def update_n_basis_modes(self, n_basis_modes, xy, **fit_kws):
        """
        Re-fit the :class:`SSPOC` object using a different value of
        :code:`n_basis_modes`.

        This method allows one to relearn sensor locations for a
        different number of basis modes _without_ re-fitting the basis
        in many cases.
        Specifically, if :code:`n_basis_modes <= self.basis.n_basis_modes`
        then the basis does not need to be refit.
        Otherwise this function does not save any computational resources.

        Parameters
        ----------
        n_basis_modes: positive int, optional (default None)
            Number of basis modes to be used during fit.
            Must be less than or equal to ``n_samples``.

        xy: tuple of np.ndarray, length 2
            Tuple containing training data x and labels y for refitting.
            x should have shape (n_samples, n_input_features) and y shape (n_samples, ).

        **fit_kws: dict, optional
            Keyword arguments to pass to :meth:`SSPOC.fit`.
        """
        if not isinstance(n_basis_modes, INT_DTYPES) or n_basis_modes <= 0:
            raise ValueError("n_basis_modes must be a positive integer")

        x, y = xy
        # No need to refit basis; only refit sensors
        if (
            hasattr(self.basis, "basis_matrix_")
            and n_basis_modes <= self.basis.n_basis_modes
        ):
            self.n_basis_modes = n_basis_modes
            self.fit(x, y, prefit_basis=True, **fit_kws)

        else:
            if n_basis_modes > x.shape[0]:
                raise ValueError(
                    "n_basis_modes cannot exceed the number of examples, x.shape[0]"
                )
            else:
                self.n_basis_modes = n_basis_modes
                self.basis.n_basis_modes = n_basis_modes
                self.fit(x, y, prefit_basis=False, **fit_kws)

    @property
    def selected_sensors(self):
        """
        Get the indices of the selected sensors.

        Returns
        -------
        sensors: numpy array, shape (n_sensors,)
            Indices of the selected sensors.
        """
        check_is_fitted(self, "sparse_sensors_")
        return self.sparse_sensors_

    def get_selected_sensors(self):
        """
        Convenience function for getting indices of the selected sensors.

        Returns
        -------
        sensors: numpy array, shape (n_sensors,)
            Indices of the selected sensors.
        """
        return self.selected_sensors
