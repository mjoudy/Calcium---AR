from .chunked_ols import solve as solve_ols
from .chunked_ridge import solve as solve_ridge


def solve_torch_normal_eq(*args, **kwargs):
    from .torch_normal_eq import solve
    return solve(*args, **kwargs)


def solve_torch_minibatch(*args, **kwargs):
    from .torch_minibatch import solve
    return solve(*args, **kwargs)


def solve_torch_linear_layer(*args, **kwargs):
    from .torch_linear_layer import solve
    return solve(*args, **kwargs)


def solve_torch_gd(*args, **kwargs):
    from .torch_gd import solve
    return solve(*args, **kwargs)


def solve_sklearn_lasso(*args, **kwargs):
    from .sklearn_lasso import solve
    return solve(*args, **kwargs)
