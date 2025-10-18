import numpy as np
from scipy.spatial import ConvexHull, QhullError
from .h_categorizer import h_categorizer


def dict_to_vector(A, d):
    """Converts a dictionary {arg: value} into a numpy vector following the order of A."""
    return np.array([d[a] for a in A], dtype=float)


def sample_and_compute_X(
    A,
    R,
    epsilon=1e-4,
    max_iter=1000,
    n_samples=10000,
    seed=42,
    controlled_args=None
):
    """Generates n_samples random weight vectors and computes corresponding h-Categorizer results."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, len(A)), dtype=float)

    for i in range(n_samples):
        w = dict(zip(A, rng.random(len(A))))

        # Override controlled arguments if specified
        if controlled_args:
            for arg, value in controlled_args.items():
                w[arg] = value

        HC = h_categorizer(A, R, w, max_iter, epsilon)
        X[i, :] = dict_to_vector(A, HC)

    return X


def _safe_hull(points, qhull_opts="QJ", jitter=1e-8):
    """
    Try to compute a convex hull robustly.
    Uses 'QJ' (joggle) and adds slight random jitter if needed.
    Returns None if still degenerate.
    """
    try:
        return ConvexHull(points, qhull_options=qhull_opts)
    except QhullError:
        try:
            pts = points + jitter * np.random.randn(*points.shape)
            return ConvexHull(pts, qhull_options=qhull_opts)
        except QhullError:
            return None


# def compute_gradual_semantics(
#     A,
#     R,
#     n_samples=1000,
#     val_axes=None,
#     controlled_args=None,
#     epsilon=1e-4,
#     max_iter=1000
# ):
#     """Compute samples and convex hull information for the given argumentation framework."""
#     X_res = sample_and_compute_X(
#         A, R, epsilon, max_iter, n_samples, controlled_args=controlled_args
#     )

#     # Case 1D
#     if len(A) == 1:
#         axes = [A[0]]
#         hull = _safe_hull(X_res)
#         dim = 1
#         return dim, axes, X_res, hull

#     # Case 2D
#     if len(A) == 2:
#         axes = A[:2]
#         hull = _safe_hull(X_res)
#         dim = 2
#         return dim, axes, X_res, hull

#     # Case ≥ 3D → project on chosen axes
#     axes = val_axes if val_axes else A[:3]
#     idx = [A.index(ax) for ax in axes]
#     Xp = X_res[:, idx]
#     hull = _safe_hull(Xp)
#     dim = 3
#     return dim, axes, Xp, hull


def compute_gradual_space(num_args, R, n_samples, axes=None, controlled_args=None, epsilon=1e-4, max_iter=1000):
    """
    Compute the convex hull (acceptability degree space) for the weighted h-categorizer.
    Returns (num_args, hull_volume, hull_area, hull_points, samples, axes)
    """
    # Generate argument labels A, B, C, ...
    A = [chr(ord("A") + i) for i in range(num_args)]

    # 1. Sample and compute semantics
    X_res = sample_and_compute_X(
        A, R, epsilon, max_iter, n_samples, controlled_args=controlled_args
    )

    # 2. Handle projections depending on argument count
    if num_args == 1:
        dim = 1
        axes_used = [A[0]]
        hull_points = np.array([[np.min(X_res)], [np.max(X_res)]])
        hull_volume = float(np.max(X_res) - np.min(X_res))
        hull_area = None
        return num_args, hull_volume, hull_area, hull_points.tolist(), X_res.tolist(), axes_used

    if num_args == 2:
        dim = 2
        axes_used = A[:2]
        hull = _safe_hull(X_res)
        if hull is None:
            hull_volume = 0.0
            hull_area = 0.0
            hull_points = []
        else:
            hull_volume = float(hull.volume)
            hull_area = float(hull.area)
            hull_points = hull.points[hull.vertices].tolist()
        return num_args, hull_volume, hull_area, hull_points, X_res.tolist(), axes_used

    # num_args >= 3
    axes_used = axes if axes else A[:3]
    idx = [A.index(ax) for ax in axes_used]
    Xp = X_res[:, idx]

    hull = _safe_hull(Xp)
    if hull is None:
        hull_volume = 0.0
        hull_area = 0.0
        hull_points = []
    else:
        hull_volume = float(hull.volume)
        hull_area = float(hull.area)
        hull_points = hull.points[hull.vertices].tolist()

    return num_args, hull_volume, hull_area, hull_points, Xp.tolist(), axes_used
