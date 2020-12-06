import warnings

from numpy import ndim
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import OrthogonalMatchingPursuit


def constrained_binary_solve(
    w, psi, quiet=False, fit_intercept=True, normalize=True, precompute="auto"
):
    if ndim(w) != 1:
        raise ValueError(
            f"w must be a 1D vector; received a vector of dimension {ndim(w)}"
        )

    model = OrthogonalMatchingPursuit(
        tol=0, fit_intercept=fit_intercept, normalize=normalize, precompute=precompute
    )

    if quiet:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            model.fit(psi, w)
    else:
        model.fit(psi, w)

    return model.coef_


def constrained_multiclass_solve(w, psi, alpha=1.0, quiet=False, **lasso_kws):
    """
    Solve

    .. math::

        \\text{argmin}_s \\|s\\|_0 \\\\
        \\text{subject to} \\|w - \\psi s\\|_2^2 \\leq tol
    """
    model = MultiTaskLasso(alpha=alpha, **lasso_kws)

    if quiet:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            model.fit(psi, w)
    else:
        model.fit(psi, w)

    return model.coef_.T
