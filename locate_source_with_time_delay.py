import numpy as np

from alignment import calculate_cross_correlation, convert_samples_to_seconds, plot_both_signals_around_max, \
    plot_n_signals_around_max, plot_two_signals_around_point, plot_both_signals, plot_n_signals, align_and_plot
from locate.find_distance import calculate_x
from locate.n_mics_intersection import calculate_n_mic_points
from recording import FS, load_n_wav_files, load_two_wav_files


def get_aligned_target_signals(signals, calib_range, target_range, fs=FS):
    n = len(signals)
    if n < 2:
        raise ValueError("Need at least 2 signals to compare.")

    # ---------------------------------------------------------
    # PHASE 1: CALIBRATION (Neutralize Hardware/Start Delays)
    # ---------------------------------------------------------
    a, b = calib_range
    calib_start = int(a * fs)
    calib_end = int(b * fs)

    # Extract the calibration window
    calib_signals = [sig[calib_start:calib_end] for sig in signals]
    plot_n_signals_around_max(calib_signals, title="Calibration Signals")
    # Calculate calibration lags relative to Mic 0
    calib_lags = [0] * n
    for i in range(1, n):
        calib_lags[i] = calculate_cross_correlation(calib_signals[0], calib_signals[i], fs=fs, verbose=False)

    # Find out how much "dead air" to chop off the start of each original signal
    min_lag = min(calib_lags)
    start_indices = [lag - min_lag for lag in calib_lags]

    # Align the FULL ORIGINAL signals
    aligned_signals = []
    for i in range(n):
        start_idx = start_indices[i]
        aligned_signals.append(signals[i][start_idx:])

    # Truncate ends so they all match the shortest array length
    min_len = min(len(sig) for sig in aligned_signals)
    aligned_signals = [sig[:min_len] for sig in aligned_signals]
    plot_n_signals_around_max(aligned_signals, title="Aligned Calibration Signals")

    print(f"--- Phase 1: Hardware Aligned ({a}s to {b}s) ---")

    # ---------------------------------------------------------
    # PHASE 2: TARGET MEASUREMENT (Calculate Arrival Times)
    # ---------------------------------------------------------
    c, d = target_range
    target_start = int(c * fs)
    target_end = int(d * fs)

    if target_end > min_len:
        target_end = min_len

    # Extract the target window from the ALIGNED signals
    target_signals = [sig[target_start:target_end] for sig in aligned_signals]
    plot_n_signals(aligned_signals, title="Target Signals After Alignment")
    plot_n_signals_around_max(target_signals, title="Target Signals After Alignment")
    return target_signals

def get_t_arrivals_from_audio(signals, calib_range, target_range, fs=FS):
    """
    1. Uses a calibration sound (mics clustered) to align hardware start times.
    2. Uses a target sound (mics spread out) to find the acoustic time-of-flight.
    3. Returns a t_arrivals dictionary perfectly formatted for TDOA math.
    """
    n = len(signals)
    target_signals = get_aligned_target_signals(signals, calib_range, target_range, fs=fs)

    # Mic 0 is our reference point in time
    t_arrivals = {0: 0.0}

    print(f"\n--- Phase 2: Acoustic Arrival Times ---")
    print("Mic 0: 0.000000s (Reference)")

    for i in range(1, n):
        # Compare Mic 0 to Mic i
        lag_samples = calculate_cross_correlation(target_signals[0], target_signals[i], fs=fs, verbose=False)
        delay_sec = convert_samples_to_seconds(lag_samples, fs)

        # If lag > 0, Mic 0 heard it first, meaning Mic i heard it LATER (positive time).
        # If lag < 0, Mic i heard it first, meaning Mic i heard it BEFORE Mic 0 (negative time).
        t_arrivals[i] = delay_sec

        print(f"Mic {i}: {delay_sec:.6f}s")

    return t_arrivals


def locate_source_from_audio(file_desc, mics_dict, calib_range, target_range, n=None, fs=44100):
    """
    End-to-end pipeline:
    1. Loads audio files.
    2. Syncs them and calculates acoustic arrival times.
    3. Runs the TDOA math to estimate the source coordinate.
    """
    print(f"\n=======================================================")
    print(f" LOCATING SOURCE: {file_desc}")
    print(f"=======================================================")

    # 1. Load the raw audio files
    signals = load_n_wav_files(file_desc, n)

    # Safety check
    if len(signals) != len(mics_dict):
        raise ValueError(f"Loaded {len(signals)} audio files, but provided {len(mics_dict)} mic coordinates.")

    # 2. Sync hardware and extract acoustic arrival times
    t_arrivals = get_t_arrivals_from_audio(signals, calib_range, target_range, fs=fs)


    # 3. Calculate intersections using your TDOA math solver
    points = calculate_n_mic_points(mics_dict, t_arrivals)

    # 4. Find the final averaged coordinate
    if points:
        est_intersection = np.mean(points, axis=0)
        print(f"\n--- RESULTS ---")
        print(f"Processed {len(points)} valid intersection points.")
        print(f"Estimated Source Location: x = {est_intersection[0]:.4f}, y = {est_intersection[1]:.4f}")
        return est_intersection, points
    else:
        print("\n--- RESULTS ---")
        print(
            "FAILED: Could not find any valid intersecting curves. Ensure mics are physically positioned correctly relative to the math model.")
        return None, None


def find_x_from_audio(sig1, sig2, d, y, fs=FS):
    """
    Loads two mic files, finds the acoustic delay, and calculates the X coordinate.
    """

    # 2. Find the time delay (t)
    _, _, lag_samples = align_and_plot(sig1, sig2)
    t = convert_samples_to_seconds(lag_samples, fs)

    print(f"Mic Distance (d): {d} m")
    print(f"Target Y (y): {y} m")
    print(f"Measured Delay (t): {t:.6f} seconds")

    if t > 0:
        print("-> Mic 1 heard the sound first.")
    elif t < 0:
        print("-> Mic 2 heard the sound first.")
    else:
        print("-> Both mics heard the sound simultaneously.")

    # 3. Calculate X
    try:
        calculated_x = calculate_x(t, d, y)
        print(f"\nCalculated X Coordinate: {calculated_x:.4f} m")
        return calculated_x
    except ValueError as e:
        print(f"\nMath Error: {e}")
        return None

def main():
    filedesc1 = "20260609_173250"
    filedesc2 = "20260609_183735"

    signals = load_two_wav_files(filedesc2)

    calib_range = (0, 10)
    target_range = (26, 29)
    d = 2
    y = 0

    #true 63
    target1, target2 = get_aligned_target_signals(signals, calib_range, target_range, fs=FS)

    find_x_from_audio(target1, target2, d, y, fs=44100)

if __name__ == "__main__":
    main()