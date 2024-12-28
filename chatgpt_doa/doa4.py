import numpy as np
from scipy.optimize import least_squares
import math

# Speed of sound
C = 343.0  # m/s

# Let d = 0.0457  # 4.57 cm
d = 0.0457

# Microphone positions (square, centered at (0,0))
# mic0 = ( d/2,  d/2)
# mic1 = (-d/2,  d/2)
# mic2 = (-d/2, -d/2)
# mic3 = ( d/2, -d/2)
mic_positions = np.array([
    [ d/2,  d/2],  # mic0
    [-d/2,  d/2],  # mic1
    [-d/2, -d/2],  # mic2
    [ d/2, -d/2],  # mic3
])

def predicted_tdoas(xy, mic_positions, speed=C):
    """
    Given a candidate source location (x,y), compute the TDOAs
    for mic1->mic0, mic2->mic0, mic3->mic0.
    Returns array [T_{1,0}, T_{2,0}, T_{3,0}].
    """
    x0, y0 = mic_positions[0]  # mic0
    d0 = np.sqrt((xy[0]-x0)**2 + (xy[1]-y0)**2)
    
    tdoas = []
    for i in [1,2,3]:
        xi, yi = mic_positions[i]
        di = np.sqrt((xy[0]-xi)**2 + (xy[1]-yi)**2)
        # T_{i,0} = (di - d0)/speed
        tdoas.append( (di - d0)/speed )
    print(tdoas)
    return np.array(tdoas)

def residuals_tdoa(xy, measured, mic_positions, speed=C):
    """
    Residual between measured TDOAs and predicted TDOAs for location xy.
    """
    pred = predicted_tdoas(xy, mic_positions, speed)
    return measured - pred

def solve_for_location(measured_tdoas, mic_positions, speed=C):
    """
    Solve for (x,y) that best matches the measured T_{1,0}, T_{2,0}, T_{3,0}.
    measured_tdoas = [T_{1,0}, T_{2,0}, T_{3,0}].
    """
    # initial guess (x,y) in front of mic0 maybe 0.2m away
    x0 = np.array([0.1, 0.1])
    res = least_squares(residuals_tdoa, x0,
                        args=(measured_tdoas, mic_positions, speed),
                        method='lm')
    return res.x  # (x*, y*)

def compute_azimuth(xy):
    """
    Convert (x,y) to azimuth in degrees [0..360).
    """
    angle = math.degrees(math.atan2(xy[1], xy[0]))
    return angle % 360

# Example usage:
if __name__ == "__main__":
    # Suppose we measured T_{1,0}=0.00010s, T_{2,0}=-0.00005s, T_{3,0}=-0.00010s
    # (just an example)
    measured_tdoas = np.array([0.00010, -0.00005, -0.00010], dtype=float)

    # Solve for location
    xy_est = solve_for_location(measured_tdoas, mic_positions, C)
    print("Estimated source location: ", xy_est)

    # Compute azimuth
    azimuth_deg = compute_azimuth(xy_est)
    print(f"Estimated azimuth: {azimuth_deg:.2f} degrees")

