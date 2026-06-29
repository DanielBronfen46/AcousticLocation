import math

# --- Constants ---
SOUND_SPEED = 343.0  # Speed of sound in m/s (at room temperature)

def calculate_x(t, d, y):
    D = t * SOUND_SPEED
    if abs(D) >= d:
        raise ValueError("tc > d. something went wrong with alignment.")
    A = d**2 - D**2
    dist_from_mic2 = (d / 2) + (D * math.sqrt(A**2 + 4 * A * y**2)) / (2 * A)

    dist_from_mic1 = d-dist_from_mic2
    return dist_from_mic1

def main():
    delay = 0.0005  # Example time delay in seconds (500 microseconds)
    d = 1
    y = 2
    solutions = calculate_x(delay)
    print(f"--- Localization Test ---")
    print(f"Mic distance: {d}m")
    print(f"Target Y: {y}m")
    print(f"Measured Delay: {delay}s")
    print(f"Calculated X: {solutions:.4f}m")
    
if __name__ == "__main__":
    main()
