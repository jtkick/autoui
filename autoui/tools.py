import numpy as np
from scipy.special import comb

def bezier_curve(control_points, n_points=100, bias=2.0):
    """
    Generates a Bézier curve with more points at the beginning.

    Parameters:
    - control_points: List of (x, y) tuples.
    - n_points: Total number of points on the curve.
    - bias: Determines how much more density there is at the start (>1 for more points at start).

    Returns:
    - np.array of shape (n_points, 2) containing (x, y) coordinates.
    """
    
    def bernstein(n, k, t):
        """Compute Bernstein polynomial at t"""
        return comb(n, k) * (t ** k) * ((1 - t) ** (n - k))

    n = len(control_points) - 1  # Degree of the curve
    
    # Generate non-uniform t values with a bias factor
    t = np.linspace(0, 1, n_points)
    t = 1 - (1 - t) ** bias  # Applying bias (e.g., squaring makes more points near 0)
    
    curve = np.zeros((n_points, 2))
    for i in range(n + 1):
        curve += np.outer(bernstein(n, i, t), control_points[i])

    return curve