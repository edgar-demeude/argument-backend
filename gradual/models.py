from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Optional


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
    num_args: int = Field(..., ge=1, le=10,
                          description="Number of arguments (|A|)")

    R: List[Tuple[str, str]
            ] = Field(..., description="Attack relations (A->B format)")

    n_samples: int = Field(
        1000, ge=10, description="Number of samples for convex hull computation")

    axes: Optional[List[str]] = Field(
        None, description="Chosen arguments for 3D plot axes (X,Y,Z)")

    controlled_args: Optional[Dict[str, float]] = Field(
        None, description="Values for non-axis arguments")


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
    axes: Optional[List[str]] = None
