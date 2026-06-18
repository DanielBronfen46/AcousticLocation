import numpy as np


c = 343 #speed of sound in m/s

def algebraic_intersection(mA, mB, mC, delta_AB, delta_AC):
    """
    Analytically solves the intersection of two hyperbolas sharing a common focus (base mic).
    mA: Base microphone [x, y]
    mB, mC: The other two microphones [x, y]
    delta_AB: Distance difference (dA - dB)
    delta_AC: Distance difference (dA - dC)
    """
    xA, yA = mA
    xB, yB = mB
    xC, yC = mC

    # Calculate squared sums for each mic (K_i = x_i^2 + y_i^2)
    kA = xA**2 + yA**2
    kB = xB**2 + yB**2
    kC = xC**2 + yC**2

    # Set up the linear system A * [x, y]^T = dA * u + v
    A = np.array([
        [2*(xB - xA), 2*(yB - yA)],
        [2*(xC - xA), 2*(yC - yA)]
    ])
    
    u = np.array([2*delta_AB, 2*delta_AC])
    v = np.array([kB - kA - delta_AB**2, kC - kA - delta_AC**2])

    # Invert matrix A to solve for x and y in terms of dA
    A_inv = np.linalg.inv(A)
    g = A_inv @ u
    h = A_inv @ v

    # Now we substitute x = g[0]*dA + h[0] and y = g[1]*dA + h[1] 
    # back into the circle equation: dA^2 = (x - xA)^2 + (y - yA)^2
    # This gives us a standard quadratic equation: a*(dA)^2 + b*dA + c = 0
    a = g[0]**2 + g[1]**2 - 1
    b = 2*g[0]*(h[0] - xA) + 2*g[1]*(h[1] - yA)
    c = (h[0] - xA)**2 + (h[1] - yA)**2

    # Solve the quadratic equation
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None # Mathematically, they don't intersect

    dA_1 = (-b + np.sqrt(discriminant)) / (2*a)
    dA_2 = (-b - np.sqrt(discriminant)) / (2*a)

    # We might get two positive roots. We calculate both points and 
    # check which one actually satisfies the original delta distances.
    best_pt = None
    min_err = float('inf')

    for dA in [dA_1, dA_2]:
        if dA > 0:
            x = g[0]*dA + h[0]
            y = g[1]*dA + h[1]
            pt = np.array([x, y])

            # Verify against original constraints
            calc_dA = np.linalg.norm(pt - mA)
            calc_dB = np.linalg.norm(pt - mB)
            calc_dC = np.linalg.norm(pt - mC)
            
            err = abs((calc_dA - calc_dB) - delta_AB) + abs((calc_dA - calc_dC) - delta_AC)
            if err < min_err:
                min_err = err
                best_pt = pt

    return best_pt

# --- Execution with your data ---
def calculate_points(mics: dict, time_difs: dict):
    t01 = time_difs['01']
    t12 = time_difs['12']
    t02 = time_difs['02']
    d01 = t01 * c
    d12 = t12 * c
    d02 = t02 * c
    pt1 = algebraic_intersection(mics[0], mics[1], mics[2], d01, d02)
    pt2 = algebraic_intersection(mics[1], mics[0], mics[2], -d01, d12)
    pt3 = algebraic_intersection(mics[2], mics[0], mics[1], -d02, -d12)
    return [pt1, pt2, pt3]


def main():
    mics = {
        0: np.array([1.5, 0]),
        1: np.array([-1.5, 0]),
        2: np.array([0, 3])
    }

    time_difs = {
        '01': 0.000816,  # t0 - t1
        '12': -0.002789-0.000816,  # t1 - t2
        '02': -0.002789  # t0 - t2
    }

    points = calculate_points(mics, time_difs)

    est_intersection = np.mean(points, axis=0)
    print("--- Analytic Intersection Points ---")
    for i, pt in enumerate(points):
        print(f"Intersection {i + 1}: x = {pt[0]:.4f}, y = {pt[1]:.4f}")
    print(f"Estimated Source Location (Average): x = {est_intersection[0]:.4f}, y = {est_intersection[1]:.4f}")

if __name__ == '__main__':
    main()