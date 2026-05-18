import csv
import os
import time


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from recording import get_two_signals, FS, USE_VOICEMEETER, DURATION


def calculate_cross_correlation(sig1 ,sig2, fs=FS):

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

    return lag_in_samples

def convert_samples_to_seconds(num_samples, fs=FS):
    return num_samples / fs


def plot_signal(signal_data, fs=FS, title="Audio Signal"):
    """
    Plots a single 1D audio signal over time.
    """
    # Create an array of time values (in seconds) that matches the length of the signal
    time_axis = np.arange(len(signal_data)) / fs

    plt.figure(figsize=(10, 4))  # Width, Height
    plt.plot(time_axis, signal_data, color='blue', alpha=0.7)

    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_both_signals(sig1, sig2, fs=FS, title="Microphone Comparison"):
    """
    Plots two signals on the same graph to visually compare them.
    """
    time_axis1 = np.arange(len(sig1)) / fs
    time_axis2 = np.arange(len(sig2)) / fs

    plt.figure(figsize=(12, 5))

    # Plot Mic 1
    plt.plot(time_axis1, sig1, label="Mic 1", color='blue', alpha=0.7)
    # Plot Mic 2
    plt.plot(time_axis2, sig2, label="Mic 2", color='red', alpha=0.7)

    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_both_signals_around_max(sig1, sig2, fs=FS, title="Microphone Comparison around Maximum", zoom_window=0.1):
    """
    Plots two signals and automatically zooms in on the loudest sound (the clap).
    zoom_window defines how many seconds before and after the clap to show.
    """
    time_axis1 = np.arange(len(sig1)) / fs
    time_axis2 = np.arange(len(sig2)) / fs

    # --- NEW: Auto-Zoom Logic ---
    # 1. Find the index of the absolute loudest sample in Mic 1
    peak_index = np.argmax(np.abs(sig1))

    # 2. Convert that index into an actual time in seconds
    peak_time = time_axis1[peak_index]

    # 3. Calculate the viewing window boundaries
    x_min = max(0, peak_time - zoom_window)  # Ensure we don't zoom past 0 seconds
    x_max = min(len(sig1) / fs, peak_time + zoom_window)

    plt.figure(figsize=(12, 5))

    # Plot Mic 1
    plt.plot(time_axis1, sig1, label="Mic 1", color='blue', alpha=0.7)
    # Plot Mic 2
    plt.plot(time_axis2, sig2, label="Mic 2", color='red', alpha=0.7)

    plt.title(title + f" (Zoomed at {peak_time:.2f}s)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Apply the zoom
    plt.xlim(x_min, x_max)

    plt.tight_layout()
    plt.show()

def align_signals(sig1, sig2, lag_in_samples):
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

    return aligned_sig1, aligned_sig2

def record_and_align(duration=DURATION):
    sig1, sig2 = get_two_signals(duration=duration)

    align_and_plot(sig1, sig2)

def align_and_plot(sig1, sig2):
    lag_in_samples = calculate_cross_correlation(sig1, sig2)




    aligned1, aligned2 = align_signals(sig1, sig2, lag_in_samples)

    plot_both_signals(sig1, sig2, title="Before Alignment")
    plot_both_signals_around_max(aligned1, aligned2, title="After Alignment around Max", zoom_window=0.02)
    plot_both_signals(aligned1, aligned2, title="After Alignment")


    save_stats(sig1, sig2, lag_in_samples)

def save_stats(sig1, sig2, lag_in_samples, fs=FS):
    lag_in_seconds = convert_samples_to_seconds(lag_in_samples, fs)
    len1 = len(sig1)
    len2 = len(sig2)
    length_diff = len1 - len2

    if USE_VOICEMEETER:
        stats_filename = f'voicemeeter_alignment_stats.csv'
    else:
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


def record_split_and_align():
    sig1, sig2 = get_two_signals(duration = 10)

    # 1. Guarantee both arrays are the exact same length before splitting
    min_len = min(len(sig1), len(sig2))
    sig1_matched = sig1[:min_len]
    sig2_matched = sig2[:min_len]

    # 2. Find the exact midpoint (using // ensures it returns a whole integer)
    midpoint = min_len // 2

    # 3. Slice the arrays into halves
    sig1a = sig1_matched[:midpoint]
    sig1b = sig1_matched[midpoint:]

    sig2a = sig2_matched[:midpoint]
    sig2b = sig2_matched[midpoint:]

    align_and_plot(sig1a, sig2a)
    align_and_plot(sig1b, sig2b)


def main():
    record_and_align(duration=60)

if __name__ == "__main__":
    main()