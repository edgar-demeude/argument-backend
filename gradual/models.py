from pydantic import BaseModel
from typing import List, Tuple, Optional

class GradualInput(BaseModel):
    """
    Input model for the Weighted h-Categorizer API.

    Attributes
    ----------
    A : List[str]
        List of argument names.
    R : List[Tuple[str, str]]
        List of attack relations between arguments.
    n_samples : int
        Number of random weight samples to generate (default: 1000).
    max_iter : int
        Maximum number of iterations for convergence (default: 1000).

    Example
    -------
        # Example JSON body for POST /gradual/compute
        {
          "A": ["A", "B", "C"],
          "R": [["A", "B"], ["B", "C"]],
          "n_samples": 500,
          "max_iter": 1000
        }
    """
    A: List[str]
    R: List[Tuple[str, str]]
    n_samples: int = 1000
    max_iter: int = 1000

class GradualOutput(BaseModel):
    """
    Output model for the Weighted h-Categorizer API.

    Attributes
    ----------
    num_args : int
        Number of arguments in the framework.
    hull_volume : Optional[float]
        Volume of the Convex Hull (None if |A| <= 1).
    hull_area : Optional[float]
        Surface area of the Convex Hull (None if |A| <= 1).
    hull_points : List[List[float]]
        Coordinates of the Convex Hull vertices.
    samples : List[List[float]]
        Sampled points (h-Categorizer outputs) used to compute the hull.

    Example
    -------
        # Example response JSON from POST /gradual/compute
        {
          "num_args": 3,
          "hull_volume": 0.018,
          "hull_area": 0.143,
          "hull_points": [
              [0.83, 0.12, 0.45],
              [0.10, 0.54, 0.92],
              [0.44, 0.80, 0.33]
          ],
          "samples": [
              [0.2, 0.3, 0.7],
              [0.6, 0.4, 0.2],
              ...
          ]
        }
    """
    num_args: int
    hull_volume: Optional[float]
    hull_area: Optional[float]
    hull_points: List[List[float]]
    samples: List[List[float]]
