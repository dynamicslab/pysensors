# import cvxopt
from sklearn.linearmodel import OrthogonalMatchingPursuit


def constrained_binary_solve(w, psi):
    pass


def constrained_multiclass_solve(
    w,
    psi,
    n_sensors=None,
    tol=None,
    fit_intercept=True,
    normalize=True,
    precompute="auto",
):
    """
    Solve
    .. math::

        \\text{argmin}_s \\|s\\|_0 \
        \\text{subject to} \\|w - psi s\\|_2^2 \\leq tol
    """
    model = OrthogonalMatchingPursuit(
        n_nonzero_coefs=n_sensors,
        tol=tol,
        fit_intercept=fit_intercept,
        normalize=normalize,
        precompute=precompute,
    )

    model.fit(psi.T, w)
    return model.coef_
