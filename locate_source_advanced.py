"""
locate_source_advanced.py

An advanced CPU-based acoustic localization pipeline.
Combines the preprocessing and cross-correlation capabilities of alignment.py
with the advanced peak selection, echo rejection, and multi-microphone self-consistency
checks of the GPU localizer pipeline.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Import sound file handling helpers
from sound_file_handling import FS, load_n_wav_files, convert_m4a_to_wav
import sound_file_handling

# Import alignment/preprocessing functions
from alignment import (
    calculate_cross_correlation,
    convert_samples_to_seconds,
    svd_denoise_signals,
    butter_bandpass_filter,
    calc_envelopes,
    normalize_signals,
    process_signals_for_correlation
)

# Import localization solvers
from locate.three_way_intersection import calculate_points, plot_three_way_hyperbolas
from locate.n_mics_intersection import calculate_n_mic_points

# ==========================================
# ADVANCED PEAK SELECTION
# ==========================================

def select_peak_advanced(cc_array, lags, expected_delay=0, max_deviation=None, favor_center=True, max_phys_delay=None):
    """
    Advanced peak selection algorithm on the CPU.
    Uses the lags array to map indices in cc_array to exact sample delays.
    
    Args:
        cc_array: 1D array of correlation magnitudes.
        lags: 1D array of sample lags corresponding to cc_array.
        expected_delay: Center index expected shift (calibration offset).
        max_deviation: Maximum allowed peak shift from expected_delay in samples.
        favor_center: Toggle echo-rejection center-favoring logic.
        max_phys_delay: Maximum physically possible acoustic delay in samples.
        
    Returns:
        tuple: (selected_delay_samples, cc_array, selected_peak_index, all_candidate_peaks)
    """
    # Find local peaks on absolute correlation magnitude
    peaks, _ = find_peaks(np.abs(cc_array))
    
    if len(peaks) == 0:
        # Fallback: pick absolute maximum index
        peak_index = np.argmax(np.abs(cc_array))
        peaks = np.array([peak_index])
    else:
        if max_deviation is not None:
            # Filter peaks to only be within max_deviation of expected_delay
            valid_mask = np.abs(lags[peaks] - expected_delay) <= max_deviation
            if np.any(valid_mask):
                peaks = peaks[valid_mask]
            else:
                # No peaks found in deviation window, select the index of the maximum value within window
                valid_indices = np.where(np.abs(lags - expected_delay) <= max_deviation)[0]
                if len(valid_indices) > 0:
                    peak_index = valid_indices[np.argmax(np.abs(cc_array[valid_indices]))]
                    peaks = np.array([peak_index])
                else:
                    # Fallback to absolute max
                    peak_index = np.argmax(np.abs(cc_array))
                    peaks = np.array([peak_index])

        peak_vals = np.abs(cc_array[peaks])
        max_peak_val = np.max(peak_vals)
        absolute_max_peak = peaks[np.argmax(peak_vals)]
        
        if favor_center and max_phys_delay is not None:
            dist_from_target = np.abs(lags[absolute_max_peak] - expected_delay)
            
            # If the max peak is physically possible (with small buffer of 10 samples)
            if dist_from_target <= max_phys_delay + 10:
                peak_index = absolute_max_peak
            else:
                # Echo/reflection fallback: find best peak (within 90% of max height) closest to expected_delay
                valid_peaks = peaks[peak_vals > 0.90 * max_peak_val]
                best_peak = valid_peaks[np.argmin(np.abs(lags[valid_peaks] - expected_delay))]
                peak_index = best_peak
        else:
            peak_index = absolute_max_peak
            
    delay_samples = lags[peak_index]
    return delay_samples, cc_array, peak_index, peaks

# ==========================================
# TARGET EVENT ISOLATION
# ==========================================

def isolate_target_event(signals, window_sec=1.0, fs=FS):
    """Finds the loudest acoustic event across all channels and slices a window around it."""
    window_samples = int(window_sec * fs)
    matrix = np.vstack(signals)
    
    # Calculate energy profile across all mics
    energy = np.sum(matrix**2, axis=0)
    
    # Locate center of the loudest event
    peak_idx = np.argmax(energy)
    
    # Extract tight boundary window
    half_window = window_samples // 2
    start_idx = max(0, peak_idx - half_window)
    end_idx = min(matrix.shape[1], peak_idx + half_window)
    
    return [sig[start_idx:end_idx] for sig in signals]

# ==========================================
# PLOTTING HELPER
# ==========================================

def plot_advanced_correlations(cc_data, max_acoustic_delay=None, fs=FS, title='Cross-Correlation vs. Delay Time'):
    """
    Plots the cross-correlation arrays and the selected TDOA peaks.
    cc_data: List of dicts with keys: 'label', 'cc', 'lags', 'raw_tdoa', 'offset'
    """
    fig, axs = plt.subplots(len(cc_data), 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title)
    
    # Handle single plot case
    if len(cc_data) == 1:
        axs = [axs]
        
    for i, data in enumerate(cc_data):
        ax = axs[i]
        cc_array = data['cc']
        lags = data['lags']
        raw_tdoa = data['raw_tdoa']
        offset = data['offset']
        
        # Represent the TRUE acoustic delay on the x-axis
        true_lags_seconds = (lags - offset) / fs
        true_delay_seconds = (raw_tdoa - offset) / fs
        
        # Sort for clean plot line
        sort_idx = np.argsort(true_lags_seconds)
        
        ax.plot(true_lags_seconds[sort_idx], cc_array[sort_idx], color='purple', alpha=0.8)
        ax.axvline(x=true_delay_seconds, color='red', linestyle='--', linewidth=2,
                   label=f'Calculated Delay: {true_delay_seconds:.5f}s')
        
        ax.set_title(data['label'])
        ax.set_ylabel("Magnitude")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="upper right")
        
        if max_acoustic_delay is not None:
            max_ac_sec = max_acoustic_delay / fs
            ax.set_xlim(-max_ac_sec * 2, max_ac_sec * 2)
            
    axs[-1].set_xlabel("Delay (seconds)")
    plt.tight_layout()
    plt.show()

# ==========================================
# ADVANCED TDOA ARRIVALS EXTRACTION
# ==========================================

def get_t_arrivals_advanced(
    signals, calib_range, target_range, mics_dict, preprocessing_parameters=None,
    fs=FS, plot=False, verbose=False, gcc_phat=False, favor_center=True, isolate_target=True
):
    """
    Extracts arrival times using advanced peak selection, echo rejection,
    and self-consistency triplet checking.
    """
    n = len(signals)
    if n != 3:
        raise ValueError("Self-consistency checks require exactly 3 microphone signals.")
        
    # --- PREPROCESSING (Optional SVD) ---
    if preprocessing_parameters and preprocessing_parameters.get('svd', False):
        signals = svd_denoise_signals(signals)
        
    # Extract parameter values for keyword passing to calculate_cross_correlation
    env = preprocessing_parameters.get('env', False) if preprocessing_parameters else False
    bandpass = preprocessing_parameters.get('bandpass', False) if preprocessing_parameters else False
    normalize = preprocessing_parameters.get('normalize', False) if preprocessing_parameters else False
    
    # --- PHASE 1: Calibration (Neutralize Hardware/Start Delays) ---
    a, b = calib_range
    calib_start, calib_end = int(a * fs), int(b * fs)
    calib_signals = [sig[calib_start:calib_end] for sig in signals]
    
    print("Calculating baseline hardware sync offsets...")
    # Calculate calibration offsets for all three pairs (0-1, 0-2, 1-2)
    offset_0_1 = calculate_cross_correlation(calib_signals[0], calib_signals[1], preprocessing_parameters, fs=fs, verbose=verbose, gcc_phat=gcc_phat, env=env, bandpass=bandpass, normalize=normalize)
    offset_0_2 = calculate_cross_correlation(calib_signals[0], calib_signals[2], preprocessing_parameters, fs=fs, verbose=verbose, gcc_phat=gcc_phat, env=env, bandpass=bandpass, normalize=normalize)
    offset_1_2 = calculate_cross_correlation(calib_signals[1], calib_signals[2], preprocessing_parameters, fs=fs, verbose=verbose, gcc_phat=gcc_phat, env=env, bandpass=bandpass, normalize=normalize)
    
    print(f"Hardware Sync Offsets -> Mic 1-2: {offset_0_1} | Mic 1-3: {offset_0_2} | Mic 2-3: {offset_1_2}")
    
    # --- PHASE 2: Target Measurement (Acoustic Time of Flight) ---
    c, d = target_range
    target_start, target_end = int(c * fs), int(d * fs)
    target_signals = [sig[target_start:target_end] for sig in signals]
    
    if isolate_target:
        print("Isolating target event window...")
        target_signals = isolate_target_event(target_signals, window_sec=1.0, fs=fs)
        
    # Calculate physical distances and max possible delay limits
    c_sound = 343.0  # Speed of sound m/s
    mics = list(mics_dict.values())
    dist_0_1 = np.linalg.norm(mics[0] - mics[1])
    dist_0_2 = np.linalg.norm(mics[0] - mics[2])
    dist_1_2 = np.linalg.norm(mics[1] - mics[2])
    
    max_phys_01 = int(dist_0_1 / c_sound * fs)
    max_phys_02 = int(dist_0_2 / c_sound * fs)
    max_phys_12 = int(dist_1_2 / c_sound * fs)
    
    # Allow 1.5x deviation bounds
    max_dev_01 = int(1.5 * max_phys_01)
    max_dev_02 = int(1.5 * max_phys_02)
    max_dev_12 = int(1.5 * max_phys_12)
    
    # Perform cross-correlations returning full spectra
    # Mic 1 vs Mic 2 (0 vs 1)
    raw_01, cc_01, lags_01 = calculate_cross_correlation(
        target_signals[0], target_signals[1], preprocessing_parameters, fs=fs, verbose=verbose,
        gcc_phat=gcc_phat, env=env, bandpass=bandpass, normalize=normalize, return_cc=True
    )
    raw_tdoa_01, cc_01, pidx_01, peaks_01 = select_peak_advanced(
        cc_01, lags_01, expected_delay=offset_0_1, max_deviation=max_dev_01, favor_center=favor_center, max_phys_delay=max_phys_01
    )
    
    # Mic 1 vs Mic 3 (0 vs 2)
    raw_02, cc_02, lags_02 = calculate_cross_correlation(
        target_signals[0], target_signals[2], preprocessing_parameters, fs=fs, verbose=verbose,
        gcc_phat=gcc_phat, env=env, bandpass=bandpass, normalize=normalize, return_cc=True
    )
    raw_tdoa_02, cc_02, pidx_02, peaks_02 = select_peak_advanced(
        cc_02, lags_02, expected_delay=offset_0_2, max_deviation=max_dev_02, favor_center=favor_center, max_phys_delay=max_phys_02
    )
    
    # Mic 2 vs Mic 3 (1 vs 2)
    raw_12, cc_12, lags_12 = calculate_cross_correlation(
        target_signals[1], target_signals[2], preprocessing_parameters, fs=fs, verbose=verbose,
        gcc_phat=gcc_phat, env=env, bandpass=bandpass, normalize=normalize, return_cc=True
    )
    raw_tdoa_12, cc_12, pidx_12, peaks_12 = select_peak_advanced(
        cc_12, lags_12, expected_delay=offset_1_2, max_deviation=max_dev_12, favor_center=favor_center, max_phys_delay=max_phys_12
    )
    
    # Compute true delays (Raw Delay minus Hardware Calibration Offset)
    true_samples_01 = raw_tdoa_01 - offset_0_1
    true_samples_02 = raw_tdoa_02 - offset_0_2
    true_samples_12 = raw_tdoa_12 - offset_1_2
    
    # --- MULTI-MIC SELF-CONSISTENCY CHECK ---
    # By definition: t(0->2) = t(0->1) + t(1->2)
    max_phys_global = max(max_phys_01, max_phys_02, max_phys_12)
    consistency_threshold = max(20, max_phys_global // 5)
    
    expected_02 = true_samples_01 + true_samples_12
    expected_01 = true_samples_02 - true_samples_12
    expected_12 = true_samples_02 - true_samples_01
    
    errors = {
        "Mic 1 to Mic 2": abs(true_samples_01 - expected_01),
        "Mic 1 to Mic 3": abs(true_samples_02 - expected_02),
        "Mic 2 to Mic 3": abs(true_samples_12 - expected_12)
    }
    
    worst_pair = max(errors, key=errors.get)
    worst_error = errors[worst_pair]
    
    print(f"[CONSISTENCY] Self-consistency errors -> "
          f"1-2: {errors['Mic 1 to Mic 2']} | "
          f"1-3: {errors['Mic 1 to Mic 3']} | "
          f"2-3: {errors['Mic 2 to Mic 3']} samples  (threshold: {consistency_threshold})")
          
    discarded_pair = None
    if worst_error > consistency_threshold:
        # --- PEAK RETRY LOOP ---
        outlier_info = {
            "Mic 1 to Mic 2": (cc_01, pidx_01, peaks_01, lags_01, offset_0_1),
            "Mic 1 to Mic 3": (cc_02, pidx_02, peaks_02, lags_02, offset_0_2),
            "Mic 2 to Mic 3": (cc_12, pidx_12, peaks_12, lags_12, offset_1_2)
        }
        
        cc_out, pidx_out, peaks_out, lags_out, offset_out = outlier_info[worst_pair]
        
        # Sort alternative peaks by magnitude (descending) excluding the selected one
        alt_peaks = [p for p in peaks_out[np.argsort(np.abs(cc_out[peaks_out]))[::-1]] if p != pidx_out]
        
        retry_success = False
        for alt_peak in alt_peaks:
            candidate_raw = lags_out[alt_peak]
            candidate_true = candidate_raw - offset_out
            
            # Recheck consistency with the candidate
            if worst_pair == "Mic 1 to Mic 2":
                cand_01, cand_02, cand_12 = candidate_true, true_samples_02, true_samples_12
            elif worst_pair == "Mic 1 to Mic 3":
                cand_01, cand_02, cand_12 = true_samples_01, candidate_true, true_samples_12
            else:
                cand_01, cand_02, cand_12 = true_samples_01, true_samples_02, candidate_true
                
            retry_err = abs(cand_02 - (cand_01 + cand_12))
            if retry_err <= consistency_threshold:
                print(f"[CONSISTENCY] '{worst_pair}' -- retrying with next-best peak: "
                      f"candidate={candidate_true} samples (error improved: {worst_error} -> {retry_err})")
                
                # Apply the corrected values
                if worst_pair == "Mic 1 to Mic 2":
                    true_samples_01 = candidate_true
                    raw_tdoa_01 = candidate_raw
                elif worst_pair == "Mic 1 to Mic 3":
                    true_samples_02 = candidate_true
                    raw_tdoa_02 = candidate_raw
                else:
                    true_samples_12 = candidate_true
                    raw_tdoa_12 = candidate_raw
                retry_success = True
                break
                
        if not retry_success:
            discarded_pair = worst_pair
            print(f"[CONSISTENCY] '{worst_pair}' -- no alternative peak worked. Discarding pair.")
            
    # Compile results
    true_sec_01 = true_samples_01 / fs
    true_sec_02 = true_samples_02 / fs
    true_sec_12 = true_samples_12 / fs
    
    results = {
        "Mic 1 to Mic 2": {"samples": true_samples_01, "seconds": true_sec_01, "raw": raw_tdoa_01, "offset": offset_0_1},
        "Mic 1 to Mic 3": {"samples": true_samples_02, "seconds": true_sec_02, "raw": raw_tdoa_02, "offset": offset_0_2},
        "Mic 2 to Mic 3": {"samples": true_samples_12, "seconds": true_sec_12, "raw": raw_tdoa_12, "offset": offset_1_2},
        "worst_pair": discarded_pair
    }
    
    if plot:
        plot_data = [
            {'label': 'Mic 1 vs Mic 2', 'cc': cc_01, 'lags': lags_01, 'raw_tdoa': raw_tdoa_01, 'offset': offset_0_1},
            {'label': 'Mic 1 vs Mic 3', 'cc': cc_02, 'lags': lags_02, 'raw_tdoa': raw_tdoa_02, 'offset': offset_0_2},
            {'label': 'Mic 2 vs Mic 3', 'cc': cc_12, 'lags': lags_12, 'raw_tdoa': raw_tdoa_12, 'offset': offset_1_2}
        ]
        plot_advanced_correlations(plot_data, max_acoustic_delay=max_phys_global, fs=fs)
        
    return results

# ==========================================
# ADVANCED PIPELINE EXECUTION
# ==========================================

def locate_source_advanced_from_audio(
    file_desc, mics_dict, calib_range, target_range, preprocessing_parameters=None,
    fs=FS, plot=False, verbose=False, gcc_phat=False, favor_center=True, isolate_target=True
):
    """
    End-to-end advanced acoustic source localization pipeline on CPU.
    """
    print(f"\n=======================================================")
    print(f" ADVANCED LOCALIZATION: {file_desc}")
    print(f"=======================================================")
    
    # 1. Load audio files
    signals = load_n_wav_files(file_desc, n=3)
    
    if len(signals) != len(mics_dict):
        raise ValueError(f"Loaded {len(signals)} files, but expected coordinates for {len(mics_dict)} mics.")
        
    # 2. Extract arrival times and cross-correlation details
    results = get_t_arrivals_advanced(
        signals, calib_range, target_range, mics_dict, preprocessing_parameters=preprocessing_parameters,
        fs=fs, plot=plot, verbose=verbose, gcc_phat=gcc_phat, favor_center=favor_center, isolate_target=isolate_target
    )
    
    print("\n--- FINAL DELAY RESULTS ---")
    worst_pair = results["worst_pair"]
    for pair, data in results.items():
        if pair == "worst_pair":
            continue
        discarded_note = " [DISCARDED via consistency]" if pair == worst_pair else ""
        print(f"{pair}: {data['samples']} samples ({data['seconds']:.6f} s){discarded_note}")
        
    # 3. Format inputs for solver
    # In locate solvers, delays are expected in negative relative arrival seconds:
    # time_difs = { '01': t0 - t1, '02': t0 - t2, '12': t1 - t2 }
    time_difs = {
        '01': -results["Mic 1 to Mic 2"]["seconds"],
        '02': -results["Mic 1 to Mic 3"]["seconds"],
        '12': -results["Mic 2 to Mic 3"]["seconds"]
    }
    
    print("\n--- LOCATING SOURCE ---")
    points = calculate_points(mics_dict, time_difs)
    
    # Select the intersection point computed purely from the two good pairs
    if worst_pair == "Mic 1 to Mic 2":
        valid_points = [points[2]] if points[2] is not None else []
    elif worst_pair == "Mic 1 to Mic 3":
        valid_points = [points[1]] if points[1] is not None else []
    elif worst_pair == "Mic 2 to Mic 3":
        valid_points = [points[0]] if points[0] is not None else []
    else:
        valid_points = [p for p in points if p is not None]
        
    if plot and valid_points:
        est = np.mean(valid_points, axis=0)
        plot_three_way_hyperbolas(mics_dict, time_difs, points=points, est_intersection=est)
        
    if valid_points:
        est = np.mean(valid_points, axis=0)
        print(f"Estimated Source Location: x = {est[0]:.4f}, y = {est[1]:.4f}")
        return est, points
    else:
        print("Failed to locate source (no valid intersections found).")
        return None, None

# ==========================================
# TESTING AND OPTIMIZATION UTILITIES
# ==========================================

def test_many_2d_locations(filedesc_list, mics_dict, calib_range, target_range, true_point, preprocessing_parameters=None, gcc_phat=False):
    distances = []
    results_messages = []

    for fd in filedesc_list:
        try:
            est_loc, _ = locate_source_advanced_from_audio(
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


def optimize_preprocessing_parameters(filedesc_list, mics_dict, calib_range, target_range, true_point):
    import csv
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
    #locate_source_advanced_from_audio(filedesc, mics_dict2, calib_range, target_range, plot=True, verbose=True, gcc_phat=gcc_phat, preprocessing_parameters=preprocessing_parameters)


    filedesc_list = [filedesc6, filedesc7, filedesc8, filedesc9, filedesc10, filedesc11]
    true_point = np.array([-3.0, 3.0])
    
    #test_many_2d_locations(filedesc_list, mics_dict2, calib_range, target_range, true_point, preprocessing_parameters, gcc_phat)
    optimize_preprocessing_parameters(filedesc_list, mics_dict2, calib_range, target_range, true_point)


def main():
    test_2d_location()


if __name__ == "__main__":
    main()
