import numpy as np
from itertools import combinations

from matplotlib import pyplot as plt

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

def calculate_n_mic_points(mics: dict, t_arrivals: dict, plot=False):
    """
    Generalized TDOA solver for N microphones.
    Evaluates every unique triplet combination of microphones.
    """
    all_points = []
    
    # Generate all unique combinations of 3 microphones
    mic_ids = list(mics.keys())
    triplets = list(combinations(mic_ids, 3))
    
    for i, j, k in triplets:
        def cap_d(d, m1, m2):
            max_d = np.linalg.norm(m1 - m2) * 0.9999
            if abs(d) > max_d:
                return np.sign(d) * max_d
            return d

        # Calculate the distance differences from the arrival times dynamically,
        # capping them to physically possible boundaries (prevents math failures on bad TDOA)
        d_ij = cap_d((t_arrivals[i] - t_arrivals[j]) * c, mics[i], mics[j])
        d_ik = cap_d((t_arrivals[i] - t_arrivals[k]) * c, mics[i], mics[k])
        d_jk = cap_d((t_arrivals[j] - t_arrivals[k]) * c, mics[j], mics[k])
        
        # Calculate the 3 intersection points for this specific triplet
        pt1 = algebraic_intersection(mics[i], mics[j], mics[k], d_ij, d_ik)
        pt2 = algebraic_intersection(mics[j], mics[i], mics[k], -d_ij, d_jk)
        pt3 = algebraic_intersection(mics[k], mics[i], mics[j], -d_ik, -d_jk)
        
        # Add to our master list, ignoring 'None' values (where curves don't cross)
        for pt in [pt1, pt2, pt3]:
            if pt is not None:
                all_points.append(pt)

    total_triplets = len(triplets)
    total_attempts = total_triplets * 3  # 3 intersection calculations per triplet
    successful_points = len(all_points)

    if plot:
        # Calculate a quick estimated intersection to feed the plotter
        est_intersection = np.mean(all_points, axis=0) if all_points else None
        plot_tdoa_hyperbolas(mics, t_arrivals, points=all_points, est_intersection=est_intersection)

    print(f"Found {successful_points} valid intersections out of {total_attempts} attempted calculations. Success rate of {successful_points/total_attempts*100:.1f}%")

    return all_points


def plot_tdoa_hyperbolas(mics, t_arrivals, points=None, est_intersection=None):
    """
    Plots the microphones, the hyperbolas representing the distance differences,
    the raw intersection points, and the final estimated source location.
    """
    # 1. Determine the boundaries of our graph based on mic locations
    all_x = [m[0] for m in mics.values()]
    all_y = [m[1] for m in mics.values()]

    if est_intersection is not None:
        all_x.append(est_intersection[0])
        all_y.append(est_intersection[1])

    # Add a 15-meter padding around our outermost points so we can see the curves
    padding = 3
    x_min, x_max = min(all_x) - padding, max(all_x) + padding
    y_min, y_max = min(all_y) - padding, max(all_y) + padding

    # 2. Create a dense 2D grid of coordinates to evaluate the hyperbola equations
    xx = np.linspace(x_min, x_max, 400)
    yy = np.linspace(y_min, y_max, 400)
    X, Y = np.meshgrid(xx, yy)

    plt.figure(figsize=(10, 8))

    # 3. Plot Mics
    for mic_id, coord in mics.items():
        plt.plot(coord[0], coord[1], 'ks', markersize=8)
        plt.text(coord[0] + 0.5, coord[1] + 0.5, f'Mic {mic_id}', fontsize=12, fontweight='bold')

    # 4. Plot Hyperbolas for every unique pair of microphones
    mic_ids = list(mics.keys())
    pairs = list(combinations(mic_ids, 2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(pairs)))

    for idx, (i, j) in enumerate(pairs):
        # Target distance difference based on our audio delays
        d_ij = (t_arrivals[i] - t_arrivals[j]) * c

        # Calculate distance from every point on our grid to Mic i and Mic j
        dist_i = np.sqrt((X - mics[i][0]) ** 2 + (Y - mics[i][1]) ** 2)
        dist_j = np.sqrt((X - mics[j][0]) ** 2 + (Y - mics[j][1]) ** 2)

        # The hyperbola exists wherever the difference in distances equals d_ij
        Z = dist_i - dist_j

        # Plot the specific contour line where Z == d_ij
        # We wrap it in a try-except just in case a curve completely misses the graphed window
        try:
            plt.contour(X, Y, Z, levels=[d_ij], colors=[colors[idx]], alpha=0.5, linewidths=2)
        except ValueError:
            pass

            # 5. Plot the raw intersection points calculated by our algebraic solver
    if points:
        pts_array = np.array(points)
        plt.scatter(pts_array[:, 0], pts_array[:, 1], c='blue', marker='x', s=60, alpha=0.7,
                    label='Calculated Intersections')
        for pt in points:
            plt.text(pt[0] + 0.1, pt[1] + 0.1, f'({pt[0]:.2f}, {pt[1]:.2f})', fontsize=9, color='blue', ha='left', va='bottom')

    # 6. Plot the final estimated source (the centroid of the intersections)
    if est_intersection is not None:
        plt.plot(est_intersection[0], est_intersection[1], 'ro', markersize=10, 
                 label=f'Estimated Source: ({est_intersection[0]:.2f}, {est_intersection[1]:.2f})')

    plt.title('TDOA Hyperbolic Intersections')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Force the X and Y axes to scale equally so our circles/curves don't warp into ovals
    plt.axis('equal')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()




def main():
    # --- Execution Example with 4 Mics ---

    mics = {
        0: np.array([-2.0, -5.0]),
        1: np.array([6.0, 3.0]),
        2: np.array([-7.0, 7.0]),
        3: np.array([5.0, -4.0])  # Added a 4th mic!
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

if __name__ == '__main__':
    main()