import numpy as np
from itertools import combinations

c = 343 # speed of sound in m/s

def algebraic_intersection(mA, mB, mC, delta_AB, delta_AC):
    # [Keep your exact original function here - no changes needed]
    xA, yA = mA
    xB, yB = mB
    xC, yC = mC

    kA = xA**2 + yA**2
    kB = xB**2 + yB**2
    kC = xC**2 + yC**2

    A = np.array([
        [2*(xB - xA), 2*(yB - yA)],
        [2*(xC - xA), 2*(yC - yA)]
    ])
    
    u = np.array([2*delta_AB, 2*delta_AC])
    v = np.array([kB - kA - delta_AB**2, kC - kA - delta_AC**2])

    # Handle collinear matrices that can't be inverted
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None

    g = A_inv @ u
    h = A_inv @ v

    a = g[0]**2 + g[1]**2 - 1
    b = 2*g[0]*(h[0] - xA) + 2*g[1]*(h[1] - yA)
    c_val = (h[0] - xA)**2 + (h[1] - yA)**2

    discriminant = b**2 - 4*a*c_val
    if discriminant < 0:
        return None 

    dA_1 = (-b + np.sqrt(discriminant)) / (2*a)
    dA_2 = (-b - np.sqrt(discriminant)) / (2*a)

    best_pt = None
    min_err = float('inf')

    for dA in [dA_1, dA_2]:
        if dA > 0:
            x = g[0]*dA + h[0]
            y = g[1]*dA + h[1]
            pt = np.array([x, y])

            calc_dA = np.linalg.norm(pt - mA)
            calc_dB = np.linalg.norm(pt - mB)
            calc_dC = np.linalg.norm(pt - mC)
            
            err = abs((calc_dA - calc_dB) - delta_AB) + abs((calc_dA - calc_dC) - delta_AC)
            if err < min_err:
                min_err = err
                best_pt = pt

    return best_pt

def calculate_n_mic_points(mics: dict, t_arrivals: dict):
    """
    Generalized TDOA solver for N microphones.
    Evaluates every unique triplet combination of microphones.
    """
    all_points = []
    
    # Generate all unique combinations of 3 microphones
    mic_ids = list(mics.keys())
    triplets = list(combinations(mic_ids, 3))
    
    for i, j, k in triplets:
        # Calculate the distance differences from the arrival times dynamically
        d_ij = (t_arrivals[i] - t_arrivals[j]) * c
        d_ik = (t_arrivals[i] - t_arrivals[k]) * c
        d_jk = (t_arrivals[j] - t_arrivals[k]) * c
        
        # Calculate the 3 intersection points for this specific triplet
        pt1 = algebraic_intersection(mics[i], mics[j], mics[k], d_ij, d_ik)
        pt2 = algebraic_intersection(mics[j], mics[i], mics[k], -d_ij, d_jk)
        pt3 = algebraic_intersection(mics[k], mics[i], mics[j], -d_ik, -d_jk)
        
        # Add to our master list, ignoring 'None' values (where curves don't cross)
        for pt in [pt1, pt2, pt3]:
            if pt is not None:
                all_points.append(pt)
                
    return all_points

# --- Execution Example with 4 Mics ---

mics = {
    0: np.array([-2.0, -5.0]),
    1: np.array([6.0, 3.0]),
    2: np.array([-7.0, 7.0]),
    3: np.array([5.0, -4.0]) # Added a 4th mic!
}

# Instead of pairs, we just supply the absolute (or relative) arrival time 
# of the sound at each microphone. 
# (These specific times were simulated to originate from coordinate [0, 0])
t_arrivals = {
    0: 0.01570,
    1: 0.01955,
    2: 0.02886,
    3: 0.01866 
}

points = calculate_n_mic_points(mics, t_arrivals)

# Calculate the overall centroid
est_intersection = np.mean(points, axis=0)

print(f"--- Processed {len(points)} valid intersection points ---")
print(f"Estimated Source Location (Average): x = {est_intersection[0]:.4f}, y = {est_intersection[1]:.4f}")