import csv
import os
from datetime import datetime

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from alignment import calculate_cross_correlation, convert_samples_to_seconds, align_two_signals, align_n_signals
from plotting_functions import plot_n_signals_around_max, plot_n_signals
from locate.find_distance import calculate_x
from locate.n_mics_intersection import calculate_n_mic_points

from sound_file_handling import FS, load_n_wav_files, load_two_wav_files

import sys
import io
from itertools import product





def get_aligned_target_signals(
    signals, calib_range, target_range, preprocessing_parameters, fs=FS, plot=False, verbose=False, gcc_phat=False):
    print("Calculating aligned target signals")

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

    aligned_calib, calib_lags = align_n_signals(
        calib_signals, plot=plot, verbose=verbose, gcc_phat=gcc_phat, preprocessing_parameters=preprocessing_parameters)

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

    if plot:
        plot_n_signals(target_signals, title="Target Signals After Calibration Alignment")
        plot_n_signals_around_max(target_signals, title="Target Signals After Calibration Alignment")
    return target_signals

def get_t_arrivals_from_audio(
    signals, calib_range, target_range, preprocessing_parameters, fs=FS, plot=False, verbose=False, gcc_phat=False):
    """
    1. Uses a calibration sound (mics clustered) to align hardware start times.
    2. Uses a target sound (mics spread out) to find the acoustic time-of-flight.
    3. Returns a t_arrivals dictionary perfectly formatted for TDOA math.
    """
    n = len(signals)
    target_signals = get_aligned_target_signals(
        signals, calib_range, target_range, fs=fs, plot=plot, verbose=verbose, gcc_phat=gcc_phat,
        preprocessing_parameters=preprocessing_parameters)

    # Mic 0 is our reference point in time
    t_arrivals = {0: 0.0}

    print(f"\nAcoustic Arrival Times ---")
    print("Mic 0: 0.000000s (Reference)")

    aligned_targets, target_lags = align_n_signals(
        target_signals, plot=plot, verbose=verbose, gcc_phat=gcc_phat, preprocessing_parameters=preprocessing_parameters)

    for i in range(n):
        delay_sec = convert_samples_to_seconds(target_lags[i], fs)
        t_arrivals[i] = delay_sec

        if i > 0:
            print(f"Mic {i}: {delay_sec:.6f}s")

    return t_arrivals


def locate_source_from_audio(
    file_desc, mics_dict, calib_range, target_range, preprocessing_parameters=None, n=None, fs=FS, plot=False, verbose=False, gcc_phat=False
    ):
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
    t_arrivals = get_t_arrivals_from_audio(
        signals, calib_range, target_range, fs=fs, plot=plot, verbose=verbose, gcc_phat=gcc_phat, preprocessing_parameters=preprocessing_parameters)

    print(f"\n=======================================================")
    print(f" CALCULATING INTERSECTION OF HYPERBOLAS: {file_desc}")
    print(f"=======================================================")
    # 3. Calculate intersections using your TDOA math solver
    points = calculate_n_mic_points(mics_dict, t_arrivals, plot=plot)

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

def find_x_from_audio(sig1, sig2, d, y, preprocessing_parameters, fs=FS, plot=True, verbose=False, gcc_phat=False):
    """
    Loads two mic files, finds the acoustic delay, and calculates the X coordinate.
    """

    # 2. Find the time delay (t)
    _, _, lag_samples = align_two_signals(sig1, sig2, preprocessing_parameters=preprocessing_parameters, plot=plot, verbose=verbose, gcc_phat=gcc_phat)
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

def load_two_signals_and_find_x(filedesc, calib_range, target_range, d, y, preprocessing_parameters, fs=FS, plot=True, verbose=False, gcc_phat=False):
    signals = load_two_wav_files(filedesc)

    target1, target2 = get_aligned_target_signals(signals, calib_range, target_range, preprocessing_parameters=preprocessing_parameters, fs=fs, plot=plot, verbose=verbose, gcc_phat=gcc_phat)

    calculated_x = find_x_from_audio(target1, target2, d, y, preprocessing_parameters=preprocessing_parameters, fs=fs, plot=plot, verbose=verbose, gcc_phat=gcc_phat)

    log_localization_results(filedesc, calib_range, target_range, d, y, gcc_phat, calculated_x)

    return calculated_x


def log_localization_results(filedesc, calib_range, target_range, d, y, gcc_phat, calculated_x,
                             filename="localization_log.csv"):
    """
    Appends the results of a localization run to a CSV file.
    Creates the file and writes headers if it doesn't exist.
    """
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as csvfile:
        fieldnames = [
            'Timestamp',
            'File_Desc',
            'Mic_Dist_d',
            'Target_Y',
            'Calib_Range',
            'Target_Range',
            'GCC_PHAT',
            'Calculated_X'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write headers only if the file is brand new
        if not file_exists:
            writer.writeheader()

        # Format the result to handle Math Errors (None types) gracefully
        if calculated_x is not None:
            x_str = f"{calculated_x:.4f}"
        else:
            x_str = "MATH_ERROR"

        writer.writerow({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'File_Desc': filedesc,
            'Mic_Dist_d': d,
            'Target_Y': y,
            'Calib_Range': str(calib_range),
            'Target_Range': str(target_range),
            'GCC_PHAT': gcc_phat,
            'Calculated_X': x_str
        })


def test_1d_location():
    filedesc4 = "20260616_141546"
    filedesc5 = "20260616_141706"
    filedesc6 = "20260616_141829"
    filedesc7 = "20260616_142003"
    filedesc8 = "20260616_142123"

    filedesc = filedesc7

    preprocessing_parameters = None

    calib_range = (0, 10)
    target_range = (20, 29)
    y = 0
    d = 2

    load_two_signals_and_find_x(filedesc, calib_range, target_range, d, y, preprocessing_parameters, plot=True, verbose=False, gcc_phat=True)

def test_2d_location():
    filedesc1 = "2026-06-16_15-42-04"
    filedesc2 = "2026-06-16_15-44-16"
    filedesc3 = "2026-06-16_15-47-41"
    filedesc4 = "2026-06-16_23-15-14"

    mics_dict1 = {
        0: np.array([0.0, 0.0]),
        1: np.array([0.0, 3.0]),
        2: np.array([3.0, 0.0])
    }

    filedesc5 = "2026-06-18_22-24-18"

    filedesc6 = "2026-06-18_22-27-22"
    filedesc7 = "2026-06-18_22-28-57"
    filedesc8 = "2026-06-18_23-12-54"
    filedesc9 = "2026-06-18_23-15-19"
    filedesc10 = "2026-06-18_23-17-26"
    filedesc11 = "2026-06-18_23-18-39"

    filedesc12 = "2026-06-18_23-25-42" # sweep - too noisy
    filedesc13 = "2026-06-18_23-27-01" # sweep - too noisy
    filedesc14 = "2026-06-18_23-30-45" # cant handle whistles, too broad of a signal

    mics_dict2 = {
        0: np.array([1.5, 0.0]),
        1: np.array([-1.5, 0.0]),
        2: np.array([0.0, 3.0])
    }

    calib_range = (0, 10)
    target_range = (20, 29)

    preprocessing_parameters = {
        'env': True,
        'bandpass': True,
        'normalize': True,
    }
    gcc_phat = False

    filedesc = filedesc6
    #locate_source_from_audio(filedesc, mics_dict2, calib_range, target_range, plot=True, verbose=True, gcc_phat=gcc_phat, preprocessing_parameters=preprocessing_parameters)


    filedesc_list = [filedesc6, filedesc7, filedesc8, filedesc9, filedesc10, filedesc11]
    true_point = np.array([-3.0, 3.0])
    
    test_many_2d_locations(filedesc_list, mics_dict2, calib_range, target_range, true_point, preprocessing_parameters, gcc_phat)
    #optimize_preprocessing_parameters(filedesc_list, mics_dict2, calib_range, target_range, true_point)


def optimize_preprocessing_parameters(filedesc_list, mics_dict, calib_range, target_range, true_point):
    import sys
    import io
    from itertools import product

    parameter_combinations = []
    for gcc, ev, bp, nm, sv in product([False, True], repeat=5):
        if gcc and ev:
            continue
        parameter_combinations.append((gcc, ev, bp, nm, sv))

    best_config = None
    min_avg_dist = float('inf')
    results_to_save = []

    print(f"\nSearching over {len(parameter_combinations)} combinations of parameters...")

    old_stdout = sys.stdout

    for gcc, ev, bp, nm, sv in parameter_combinations:
        # Suppress outputs during search iteration
        sys.stdout = io.StringIO()
        pre_params = {
            'env': ev,
            'bandpass': bp,
            'normalize': nm,
            'svd': sv,
        }
        try:
            _, avg_dist, _ = test_many_2d_locations(
                filedesc_list, mics_dict, calib_range, target_range, true_point, pre_params, gcc
            )
        except Exception:
            avg_dist = None
        
        sys.stdout = old_stdout

        if avg_dist is not None:
            print(f"gcc_phat={gcc}, env={ev}, bandpass={bp}, normalize={nm}, svd={sv} -> Avg Dist = {avg_dist:.4f} m")
            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                best_config = (gcc, ev, bp, nm, sv)
        else:
            print(f"gcc_phat={gcc}, env={ev}, bandpass={bp}, normalize={nm}, svd={sv} -> Avg Dist = FAILED")

        results_to_save.append({
            'gcc_phat': gcc,
            'env': ev,
            'bandpass': bp,
            'normalize': nm,
            'svd': sv,
            'avg_dist': f"{avg_dist:.6f}" if avg_dist is not None else "FAILED"
        })

    # Save to file
    filename = "optimization_results.csv"
    with open(filename, mode='w', newline='') as csvfile:
        fieldnames = ['gcc_phat', 'env', 'bandpass', 'normalize', 'svd', 'avg_dist']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_to_save)
    print(f"\nSaved all optimization results to '{filename}'.")

    print("\n=======================================================")
    print(" BEST PARAMETERS FOUND")
    print("=======================================================")
    if best_config is not None:
        gcc, ev, bp, nm, sv = best_config
        print(f"gcc_phat  : {gcc}")
        print(f"env       : {ev}")
        print(f"bandpass  : {bp}")
        print(f"normalize : {nm}")
        print(f"svd       : {sv}")
        print(f"Minimal Avg Distance: {min_avg_dist:.4f} m")
        print("=======================================================")

        # Run and print the full details of the best configuration
        print("\nDetail of the best configuration:")
        pre_params = {
            'env': ev,
            'bandpass': bp,
            'normalize': nm,
            'svd': sv,
        }
        test_many_2d_locations(
            filedesc_list, mics_dict, calib_range, target_range, true_point, pre_params, gcc
        )
    else:
        print("No valid configurations found.")
        print("=======================================================")


def test_many_2d_locations(filedesc_list, mics_dict, calib_range, target_range, true_point, preprocessing_parameters=None, gcc_phat=False):
    distances = []
    results_messages = []

    for fd in filedesc_list:
        try:
            est_loc, _ = locate_source_from_audio(
                fd, mics_dict, calib_range, target_range, preprocessing_parameters=preprocessing_parameters, plot=False, verbose=True, gcc_phat=gcc_phat
            )
            if est_loc is not None:
                dist = np.linalg.norm(est_loc - true_point)
                distances.append(dist)
                results_messages.append(f"File {fd}: Estimated point = [{est_loc[0]:.4f}, {est_loc[1]:.4f}], Estimated Distance = {dist:.4f} m")
            else:
                results_messages.append(f"File {fd}: Estimated point = FAILED, Estimated Distance = N/A")
        except Exception as e:
            results_messages.append(f"File {fd}: Estimated point = ERROR ({e}), Estimated Distance = N/A")

    print("\n=======================================================")
    print(" SUMMARY OF ACCURACY RESULTS")
    print("=======================================================")
    for msg in results_messages:
        print(msg)

    if distances:
        avg_dist = np.mean(distances)
        std_dist = np.std(distances)
        print(f"\nAverage Distance: {avg_dist:.4f} m")
        print(f"Standard Deviation of Distance: {std_dist:.4f} m")
        print("=======================================================")
        return distances, avg_dist, std_dist
    print("\nNo locations were successfully estimated.")
    print("=======================================================")
    return distances, None, None


    


def main():
    test_2d_location()


if __name__ == "__main__":
    main()