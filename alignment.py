import csv
import os
import time


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import hilbert, correlate, correlation_lags

from plotting_functions import plot_both_signals, plot_both_signals_around_max, compare_two_signals_at_multiple_points, \
    plot_n_signals, compare_n_signals_at_multiple_points, plot_n_signals_around_max, plot_before_after_comparison
from sound_file_handling import FS, load_two_wav_files



def _calculate_cross_correlation_with_gcc(sig1, sig2, fs=FS, window_sec=0.05, verbose=False):
    """
    Two-Stage Alignment:
    1. Coarse alignment using Hilbert envelopes (fixes massive hardware sync issues).
    2. Fine alignment using GCC-PHAT on a single isolated clap.
    """

    # ==========================================
    # STAGE 1: COARSE ALIGNMENT (Envelope)
    # ==========================================
    # Get the "shape" of the audio, ignoring the high-frequency sound waves
    env1 = np.abs(hilbert(sig1))
    env2 = np.abs(hilbert(sig2))

    # Standard correlation on the envelopes to find the macro hardware delay
    coarse_corr = correlate(env2, env1, mode='full', method='fft')
    lags = correlation_lags(len(env1), len(env2), mode='full')
    coarse_lag = lags[np.argmax(coarse_corr)]

    if verbose:
        print(f"-> Stage 1 (Cross Correlation): Shifted by {coarse_lag} samples ({coarse_lag/fs:.4f}s)")

    # Temporarily shift sig2 by the coarse lag so the claps physically overlap
    if coarse_lag > 0:
        sig1_aligned = sig1
        sig2_aligned = sig2[coarse_lag:]
    elif coarse_lag < 0:
        sig1_aligned = sig1[abs(coarse_lag):]
        sig2_aligned = sig2
    else:
        sig1_aligned = sig1
        sig2_aligned = sig2

    # Make them equal length for Stage 2
    min_len = min(len(sig1_aligned), len(sig2_aligned))
    sig1_aligned = sig1_aligned[:min_len]
    sig2_aligned = sig2_aligned[:min_len]


    # ==========================================
    # STAGE 2: FINE ALIGNMENT (GCC-PHAT)
    # ==========================================
    # Now that the claps overlap, find the absolute loudest peak
    peak_idx = np.argmax(np.abs(sig1_aligned))

    # Crop a very tight window (e.g., 0.05s) around that single peak
    half_window = int(window_sec * fs)
    start_idx = max(0, peak_idx - half_window)
    end_idx = min(len(sig1_aligned), peak_idx + half_window)

    sig1_crop = sig1_aligned[start_idx:end_idx]
    sig2_crop = sig2_aligned[start_idx:end_idx]

    # Run GCC-PHAT on this tiny, single-clap window
    n = len(sig1_crop) + len(sig2_crop) - 1
    n_fft = 1 << (n - 1).bit_length()

    SIG1 = fft(sig1_crop, n=n_fft)
    SIG2 = fft(sig2_crop, n=n_fft)

    R = SIG1 * np.conj(SIG2)
    magnitude = np.abs(R)
    threshold = np.max(magnitude) * 1e-3

    cc = np.real(ifft(R / np.maximum(magnitude, threshold)))
    cc = np.concatenate((cc[-n_fft // 2:], cc[:n_fft // 2]))

    fine_lag = np.argmax(cc) - (n_fft // 2)

    if verbose:
        print(f"-> Stage 2 (GCC-PHAT): Sub-adjusted by {fine_lag} samples ({fine_lag/fs:.6f}s)")

    # ==========================================
    # COMBINE LAGS
    # ==========================================
    # The total true delay is the massive hardware shift PLUS the micro acoustic shift
    total_lag = coarse_lag + fine_lag

    save_stats(sig1, sig2, total_lag)

    return total_lag

def _crop_around_max_and_correlate(sig1, sig2, fs=FS, window_sec=0.05, verbose=False, gcc_phat=False):
    """
    Crops both signals around their independent maximums, passes the clean
    crops to the existing cross-correlation function, and calculates the true lag.
    """
    half_window = int(window_sec * fs)

    # 1. Crop Mic 1 around its absolute maximum
    peak1_idx = np.argmax(np.abs(sig1))
    start1 = max(0, peak1_idx - half_window)
    end1 = min(len(sig1), peak1_idx + half_window)
    crop1 = sig1[start1:end1]

    # 2. Crop Mic 2 around its absolute maximum
    peak2_idx = np.argmax(np.abs(sig2))
    start2 = max(0, peak2_idx - half_window)
    end2 = min(len(sig2), peak2_idx + half_window)
    crop2 = sig2[start2:end2]

    # 3. Pass the clean, single-clap crops to your existing function
    # (Assuming calculate_cross_correlation is already imported)
    # We pass gcc_phat=False or True based on which method you currently want to test
    crop_lag = calculate_cross_correlation(crop1, crop2, fs=fs, verbose=verbose, gcc_phat=gcc_phat)

    # 4. Reconstruct the global delay
    # The lag returned is the micro-delay between the two crops.
    # To get the true delay between the original 30-second files,
    # we must add the difference in where we started cutting.
    index_difference = start2 - start1
    true_lag = crop_lag + index_difference

    if verbose:
        print("-" * 30)
        print("CROPPING STATS:")
        print(f"Mic 1 cut at index: {start1}")
        print(f"Mic 2 cut at index: {start2}")
        print(f"Cut Difference: {index_difference} samples")
        print(f"Crop-to-Crop Lag: {crop_lag} samples")
        print(f"-> Final True Lag: {true_lag} samples ({true_lag/fs:.5f}s)")
        print("-" * 30)

    return true_lag

def calculate_cross_correlation(sig1 ,sig2, fs=FS, verbose=False, gcc_phat=False):

    if gcc_phat:
        return _calculate_cross_correlation_with_gcc(sig1, sig2, verbose=verbose)

    # 1. Perform the cross-correlation
    # We use method='fft' because doing this in the time domain would take forever
    correlation = signal.correlate(sig2, sig1, mode='full', method='fft')

    # 2. Generate an array of lags (the possible sample shifts)
    lags = signal.correlation_lags(len(sig2), len(sig1), mode='full')

    # 3. Find the exact lag where the two signals are most similar (the clap)
    # We use np.abs() just in case one microphone's polarity is inverted
    max_corr_idx = np.argmax(np.abs(correlation))
    lag_in_samples = lags[max_corr_idx]

    # 4. Convert the sample delay into an actual time delay in seconds
    delay_in_seconds = convert_samples_to_seconds(lag_in_samples, fs)

    if verbose:

        print("-" * 30)
        print(f"Delay in samples: {lag_in_samples}")
        print(f"Delay in seconds: {delay_in_seconds:.5f}s")

        # 5. Interpret the results
        if lag_in_samples > 0:
            print("Result: Mic 1 heard the clap first.")
        elif lag_in_samples < 0:
            print("Result: Mic 2 heard the clap first.")
        else:
            print("Result: Incredible! The microphones are perfectly aligned.")
        print(f"len_sig1: {len(sig1)}, len_sig2: {len(sig2)}")
        print("-" * 30)

        lags_in_seconds = lags / fs

        # We plot the absolute value of the correlation to clearly see the magnitude peak
        plt.figure(figsize=(10, 4))
        plt.plot(lags_in_seconds, np.abs(correlation), color='purple', alpha=0.8)

        # Draw a vertical dashed red line exactly where the maximum peak occurs
        plt.axvline(x=delay_in_seconds, color='red', linestyle='--', linewidth=2,
                    label=f'Calculated Delay: {delay_in_seconds:.5f}s')

        plt.title("Cross-Correlation vs. Delay Time")
        plt.xlabel("Delay (seconds)")
        plt.ylabel("Correlation Magnitude")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    save_stats(sig1, sig2, lag_in_samples)

    return lag_in_samples



def convert_samples_to_seconds(num_samples, fs=FS):
    return num_samples / fs

def align_signals_given_lag(sig1, sig2, lag_in_samples):
    """
    Aligns two signals based on the calculated lag and truncates
    them so they are exactly the same length. Returns the aligned numpy arrays.
    """

    # 1. Align the starts of the arrays based on the time delay
    if lag_in_samples > 0:
        # Mic 1 heard it first. Mic 2 is delayed and has extra "dead air" at the start.
        # We must chop off the beginning of Mic 2 to shift it backwards in time.
        aligned_sig1 = sig1
        aligned_sig2 = sig2[lag_in_samples:]
    elif lag_in_samples < 0:
        # Mic 2 heard it first. Mic 1 is delayed.
        # We must chop off the beginning of Mic 1.
        abs_lag = abs(lag_in_samples)
        aligned_sig1 = sig1[abs_lag:]
        aligned_sig2 = sig2
    else:
        aligned_sig1 = sig1
        aligned_sig2 = sig2

    # 2. Truncate the ends so both arrays are the exact same length
    min_len = min(len(aligned_sig1), len(aligned_sig2))
    aligned_sig1 = aligned_sig1[:min_len]
    aligned_sig2 = aligned_sig2[:min_len]

    #aligned_sig1, aligned_sig2 = trim_zeroes(aligned_sig1, aligned_sig2)

    return aligned_sig1, aligned_sig2

def align_two_signals(sig1, sig2, plot=True, verbose=False, gcc_phat=False):
    lag_in_samples = calculate_cross_correlation(sig1, sig2, verbose=verbose, gcc_phat=gcc_phat)

    aligned1, aligned2 = align_signals_given_lag(sig1, sig2, lag_in_samples)

    if plot:
        signals = [sig1, sig2]
        aligned = [aligned1, aligned2]
        plot_before_after_comparison(signals, aligned)
        compare_two_signals_at_multiple_points(aligned1, aligned2, n_points=6)

    return aligned1, aligned2, lag_in_samples

def calculate_lags_and_align_n_signals(signals, fs=FS, verbose=False, gcc_phat=False):
    """
    Aligns a list of N signals using signals[0] as the reference.
    """
    n = len(signals)
    lags = [0] * n  # Reference has 0 lag to itself

    # 1. Calculate lags relative to the first signal
    for i in range(1, n):
        print(f"CALCULATING LAG OF MIC {i}:")
        # We set verbose=False so it calculates quietly in the background
        lags[i] = calculate_cross_correlation(signals[0], signals[i], fs=fs, verbose=verbose, gcc_phat=gcc_phat)

    # 2. Normalize the lags into start indices
    # A negative lag means a signal heard the sound EARLIER than the reference.
    # By subtracting the minimum lag, we find out how much "dead air" to chop off each start.
    min_lag = min(lags)
    start_indices = [lag - min_lag for lag in lags]

    # 3. Slice the beginnings to align them
    aligned_signals = []
    for i in range(n):
        start_idx = start_indices[i]
        aligned_signals.append(signals[i][start_idx:])

    # 4. Truncate ends so they all match the shortest array length
    min_len = min(len(sig) for sig in aligned_signals)
    aligned_signals = [sig[:min_len] for sig in aligned_signals]

    # Note: If your imported `trim_zeroes` only handles 2 inputs, you might need to
    # write a new one for a list of arrays, or simply skip it for now.

    return aligned_signals, lags

def align_n_signals(signals, plot=True, verbose=False, gcc_phat=False):

    aligned_signals, lags = calculate_lags_and_align_n_signals(signals, verbose=verbose, gcc_phat=gcc_phat)

    if plot:
        plot_before_after_comparison(signals, aligned_signals)
        compare_n_signals_at_multiple_points(aligned_signals, n_points=4)

    return aligned_signals, lags


def save_stats(sig1, sig2, lag_in_samples, fs=FS):
    lag_in_seconds = convert_samples_to_seconds(lag_in_samples, fs)
    len1 = len(sig1)
    len2 = len(sig2)
    length_diff = len1 - len2

    stats_filename = f'direct_alignment_stats.csv'

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(stats_filename)

    # Open in append mode ('a'). newline='' is required for Windows CSV writing.
    with open(stats_filename, 'a', newline='') as csvfile:
        fieldnames = ['Timestamp', 'Length_Sig1', 'Length_Sig2', 'Length_Diff', 'Lag_Samples', 'Lag_Seconds']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row only if this is a brand new file
        if not file_exists:
            writer.writeheader()

        # Write the data for this specific run
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow({
            'Timestamp': current_time,
            'Length_Sig1': len1,
            'Length_Sig2': len2,
            'Length_Diff': length_diff,
            'Lag_Samples': lag_in_samples,
            'Lag_Seconds': f"{lag_in_seconds:.7f}"
        })

    print(f"Appended alignment statistics to {stats_filename}")

def load_and_align(file_desc):
    sig1, sig2 = load_two_wav_files(file_desc)

    align_two_signals(sig1, sig2)


def main():
    filedesc1 = "20260609_173250"
    filedesc2 = "20260609_180357"
    load_and_align(filedesc2)

if __name__ == "__main__":
    main()