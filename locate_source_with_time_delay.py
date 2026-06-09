import numpy as np

from alignment import calculate_cross_correlation, convert_samples_to_seconds
from locate.n_mics_intersection import calculate_n_mic_points
from recording import FS, load_n_wav_files


def get_t_arrivals_from_audio(signals, calib_range, target_range, fs=FS):
    """
    1. Uses a calibration sound (mics clustered) to align hardware start times.
    2. Uses a target sound (mics spread out) to find the acoustic time-of-flight.
    3. Returns a t_arrivals dictionary perfectly formatted for TDOA math.
    """
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

    # Mic 0 is our reference point in time
    t_arrivals = {0: 0.0}

    print(f"\n--- Phase 2: Acoustic Arrival Times ({c}s to {d}s) ---")
    print("Mic 0: 0.000000s (Reference)")

    for i in range(1, n):
        # Compare Mic 0 to Mic i
        lag_samples = calculate_cross_correlation(target_signals[0], target_signals[i], fs=fs, verbose=False)
        delay_sec = convert_samples_to_seconds(lag_samples, fs)

        # If lag > 0, Mic 0 heard it first, meaning Mic i heard it LATER (positive time).
        # If lag < 0, Mic i heard it first, meaning Mic i heard it BEFORE Mic 0 (negative time).
        t_arrivals[i] = delay_sec

        print(f"Mic {i}: {delay_sec:.6f}s")

    return t_arrivals, aligned_signals


def locate_source_from_audio(file_desc, mics_dict, calib_range, target_range, n=None, fs=48000):
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
    t_arrivals, synced_signals = get_t_arrivals_from_audio(signals, calib_range, target_range, fs=fs)


    # 3. Calculate intersections using your TDOA math solver
    points = calculate_n_mic_points(mics_dict, t_arrivals)

    # 4. Find the final averaged coordinate
    if points:
        est_intersection = np.mean(points, axis=0)
        print(f"\n--- RESULTS ---")
        print(f"Processed {len(points)} valid intersection points.")
        print(f"Estimated Source Location: x = {est_intersection[0]:.4f}, y = {est_intersection[1]:.4f}")
        return est_intersection, points, synced_signals
    else:
        print("\n--- RESULTS ---")
        print(
            "FAILED: Could not find any valid intersecting curves. Ensure mics are physically positioned correctly relative to the math model.")
        return None, None, synced_signals