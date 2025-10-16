from scipy.spatial import ConvexHull
import numpy as np
from .h_categorizer import sample_and_compute_X

def compute_gradual_semantics(A, R, n_samples=1000, max_iter=1000):
    """Compute samples and convex hull information for the given argumentation framework."""
    X_res = sample_and_compute_X(A, R, max_iter=max_iter, n_samples=n_samples)
    result = {"num_args": len(A)}

    if len(A) > 1:
        hull = ConvexHull(X_res)
        result["hull_volume"] = float(hull.volume)
        result["hull_area"] = float(hull.area)
        result["hull_points"] = hull.points[hull.vertices].tolist()
    else:
        result["hull_volume"] = None
        result["hull_area"] = None
        result["hull_points"] = X_res.tolist()

    result["samples"] = X_res.tolist()
    return result
