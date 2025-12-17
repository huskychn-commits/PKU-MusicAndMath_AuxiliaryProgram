"""
Tube shape function (differentiable)
Input: x (a number in [0,1] range)
Output: corresponding Y coordinate
Raises exception if input is outside [0,1] range
The function is differentiable using spline interpolation (order=1).
"""

import numpy as np

def tube_shape(x):
    """
    Compute tube shape function (differentiable)
    
    Parameters:
    x -- normalized coordinate, must be in [0,1] range
    
    Returns:
    y -- corresponding Y coordinate
    
    Raises:
    ValueError -- if x is not in [0,1] range
    ImportError -- if scipy is not available (required for differentiable interpolation)
    """
    # Check input range
    if x < 0 or x > 1:
        raise ValueError(f"Input x={x} is outside [0,1] range")
    
    # Original data points
    x_vals = [0.13043478260869565, 0.5246376811594203]
    y_vals = [0.28923076923076924, 0.5384615384615384]
    
    # X coordinate range
    x_min = 0.13043478260869565
    x_max = 0.5246376811594203
    
    # Number of points and spline order
    n_points = 2
    spline_order = 1
    
    # For differentiable interpolation, we require scipy
    try:
        from scipy import interpolate
    except ImportError:
        raise ImportError(
            "scipy is required for differentiable tube_shape function. "
            "Please install scipy: pip install scipy"
        )
    
    # Create spline interpolation with appropriate order
    # For linear interpolation (k=1), we need at least 2 points
    # For quadratic spline (k=2), we need at least 3 points
    # For cubic spline (k=3), we need at least 4 points
    tck = interpolate.splrep(x_vals, y_vals, s=0, k=spline_order)
    
    # Map input x from [0,1] to actual X coordinate range
    x_scaled = x_min + x * (x_max - x_min)
    
    # Ensure x_scaled is within the interpolation range
    # (splev can extrapolate, but we want to stay within bounds)
    x_scaled = max(min(x_scaled, x_vals[-1]), x_vals[0])
    
    # Compute interpolation using spline
    y = interpolate.splev(x_scaled, tck, der=0)
    return float(y)

def tube_shape_derivative(x, order=1):
    """
    Compute derivative of tube shape function
    
    Parameters:
    x -- normalized coordinate, must be in [0,1] range
    order -- order of derivative (1 for first derivative, 2 for second derivative)
    
    Returns:
    dy/dx -- derivative of tube shape function at x
    
    Raises:
    ValueError -- if x is not in [0,1] range or order is not 1 or 2
    ImportError -- if scipy is not available
    """
    if x < 0 or x > 1:
        raise ValueError(f"Input x={x} is outside [0,1] range")
    
    if order not in [1, 2]:
        raise ValueError(f"Derivative order must be 1 or 2, got {order}")
    
    # Original data points
    x_vals = [0.13043478260869565, 0.5246376811594203]
    y_vals = [0.28923076923076924, 0.5384615384615384]
    
    # X coordinate range
    x_min = 0.13043478260869565
    x_max = 0.5246376811594203
    
    # Number of points and spline order
    n_points = 2
    spline_order = 1
    
    # Check if derivative is available for the given spline order
    if spline_order == 1 and order > 1:
        raise ValueError(f"Cannot compute order {order} derivative for linear interpolation (spline_order=1)")
    if spline_order == 2 and order > 2:
        raise ValueError(f"Cannot compute order {order} derivative for quadratic spline (spline_order=2)")
    
    # Require scipy for derivative computation
    try:
        from scipy import interpolate
    except ImportError:
        raise ImportError(
            "scipy is required for derivative computation. "
            "Please install scipy: pip install scipy"
        )
    
    # Create spline interpolation with appropriate order
    tck = interpolate.splrep(x_vals, y_vals, s=0, k=spline_order)
    
    # Map input x from [0,1] to actual X coordinate range
    x_scaled = x_min + x * (x_max - x_min)
    
    # Ensure x_scaled is within the interpolation range
    x_scaled = max(min(x_scaled, x_vals[-1]), x_vals[0])
    
    # Compute derivative using spline
    derivative = interpolate.splev(x_scaled, tck, der=order)
    return float(derivative)

# Example usage and testing
if __name__ == "__main__":
    # Test the function and its derivatives
    test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print("Testing tube_shape function (differentiable):")
    for x in test_points:
        try:
            y = tube_shape(x)
            print(f"  tube_shape({x}) = {y:.6f}")
        except (ValueError, ImportError) as e:
            print(f"  tube_shape({x}) error: {e}")
    
    print("\nTesting first derivative:")
    for x in test_points:
        try:
            dy = tube_shape_derivative(x, order=1)
            print(f"  tube_shape'({x}) = {dy:.6f}")
        except (ValueError, ImportError) as e:
            print(f"  tube_shape'({x}) error: {e}")
    
    print("\nTesting second derivative:")
    for x in test_points:
        try:
            d2y = tube_shape_derivative(x, order=2)
            print(f"  tube_shape''({x}) = {d2y:.6f}")
        except (ValueError, ImportError) as e:
            print(f"  tube_shape''({x}) error: {e}")
