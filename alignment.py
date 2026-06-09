import csv
import os
import time


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from recording import record_two_signals, FS, trim_zeroes, load_two_wav_files


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

def plot_two_signals_around_point(sig1, sig2, point_time, fs=FS, title=f"Zoomed in Microphone Comparison", zoom_window=0.1):
    time_axis1 = np.arange(len(sig1)) / fs
    time_axis2 = np.arange(len(sig2)) / fs

    # 3. Calculate the viewing window boundaries
    x_min = max(0, point_time - zoom_window)  # Ensure we don't zoom past 0 seconds
    x_max = min(len(sig1) / fs, point_time + zoom_window)

    plt.figure(figsize=(12, 5))

    # Plot Mic 1
    plt.plot(time_axis1, sig1, label="Mic 1", color='blue', alpha=0.7)
    # Plot Mic 2
    plt.plot(time_axis2, sig2, label="Mic 2", color='red', alpha=0.7)

    plt.title(title + f" (Zoomed at {point_time:.2f}s)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Apply the zoom
    plt.xlim(x_min, x_max)

    plt.tight_layout()
    plt.show()

def compare_two_signals_at_multiple_points(sig1, sig2, n_points, fs=FS, zoom_window=0.1):
    time_len = len(sig1) / fs
    point_arr = np.linspace(zoom_window, time_len-zoom_window, num=n_points, endpoint=True)

    for i in range(len(point_arr)):
        plot_two_signals_around_point(sig1, sig2, point_arr[i], fs=fs, title=f"Microphone Comparison {i+1}/{n_points}")

def plot_n_signals(signals, fs=FS, title="N Microphones Comparison"):
    plt.figure(figsize=(12, 5))

    for i, sig in enumerate(signals):
        time_axis = np.arange(len(sig)) / fs
        plt.plot(time_axis, sig, label=f"Mic {i + 1}", alpha=0.7)

    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_n_signals_around_max(signals, fs=FS, title="Zoomed Microphones", zoom_window=0.1):
    plt.figure(figsize=(12, 5))

    # Use the first signal to find the peak for zooming
    peak_index = np.argmax(np.abs(signals[0]))
    peak_time = peak_index / fs
    x_min = max(0, peak_time - zoom_window)
    x_max = min(len(signals[0]) / fs, peak_time + zoom_window)

    for i, sig in enumerate(signals):
        time_axis = np.arange(len(sig)) / fs
        plt.plot(time_axis, sig, label=f"Mic {i + 1}", alpha=0.7)

    plt.title(f"{title} (Zoomed at {peak_time:.2f}s)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.xlim(x_min, x_max)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_n_signals_around_point(signals, point_time, fs=FS, title="Zoomed N-Microphone Comparison", zoom_window=0.1):
    """
    Plots a list of N signals zoomed in around a specific point in time.
    """
    if not signals:
        return

    # Calculate the viewing window boundaries based on the target point
    x_min = max(0, point_time - zoom_window)
    # Use the first signal's length as a safety cap for the max x-axis
    max_time = len(signals[0]) / fs
    x_max = min(max_time, point_time + zoom_window)

    plt.figure(figsize=(12, 5))

    # Plot every signal dynamically
    for i, sig in enumerate(signals):
        time_axis = np.arange(len(sig)) / fs
        plt.plot(time_axis, sig, label=f"Mic {i+1}", alpha=0.7)

    plt.title(title + f" (Zoomed at {point_time:.2f}s)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Apply the zoom
    plt.xlim(x_min, x_max)

    plt.tight_layout()
    plt.show()

def compare_n_signals_at_multiple_points(signals, n_points, fs=FS, zoom_window=0.1):
    """
    Calculates evenly spaced points across N signals and plots zoomed-in comparisons for each.
    """
    if not signals:
        return

    # Use the length of the first signal to represent the total time.
    # (Assuming they have already been aligned and truncated to the same length)
    time_len = len(signals[0]) / fs

    # Generate the evenly spaced points
    point_arr = np.linspace(zoom_window, time_len - zoom_window, num=n_points, endpoint=True)

    # Loop through each calculated point and plot
    for i, point_time in enumerate(point_arr):
        plot_title = f"N-Microphone Comparison {i + 1}/{n_points}"
        plot_n_signals_around_point(
            signals,
            point_time,
            fs=fs,
            title=plot_title,
            zoom_window=zoom_window
        )


def calculate_cross_correlation(sig1 ,sig2, fs=FS, verbose=True):

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

    aligned_sig1, aligned_sig2 = trim_zeroes(aligned_sig1, aligned_sig2)

    return aligned_sig1, aligned_sig2

def align_and_plot(sig1, sig2):
    lag_in_samples = calculate_cross_correlation(sig1, sig2)

    aligned1, aligned2 = align_signals_given_lag(sig1, sig2, lag_in_samples)

    plot_both_signals(sig1, sig2, title="Before Alignment")
    plot_both_signals(aligned1, aligned2, title="After Alignment")
    plot_both_signals_around_max(aligned1, aligned2, title="After Alignment around Max", zoom_window=0.02)
    compare_two_signals_at_multiple_points(aligned1, aligned2, n_points=6)

    return aligned1, aligned2, lag_in_samples

def align_n_signals(signals, fs=FS):
    """
    Aligns a list of N signals using signals[0] as the reference.
    """
    n = len(signals)
    lags = [0] * n  # Reference has 0 lag to itself

    # 1. Calculate lags relative to the first signal
    for i in range(1, n):
        # We set verbose=False so it calculates quietly in the background
        lags[i] = calculate_cross_correlation(signals[0], signals[i], fs=fs, verbose=False)

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

def align_and_plot_n(signals):

    aligned_signals, lags = align_n_signals(signals)

    plot_n_signals(signals, title="Before Alignment")
    plot_n_signals(aligned_signals, title="After Alignment")
    plot_n_signals_around_max(aligned_signals, title="After Alignment around Max", zoom_window=0.02)
    compare_n_signals_at_multiple_points(aligned_signals, n_points=4)

    return aligned_signals, lags


def record_and_align(duration):
    sig1, sig2 = record_two_signals(duration=duration)

    align_and_plot(sig1, sig2)

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

def record_split_and_align():
    sig1_matched, sig2_matched = record_two_signals(duration = 10)

    min_len = min(len(sig1_matched), len(sig2_matched))

    # 2. Find the exact midpoint (using // ensures it returns a whole integer)
    midpoint = min_len // 2

    # 3. Slice the arrays into halves
    sig1a = sig1_matched[:midpoint]
    sig1b = sig1_matched[midpoint:]

    sig2a = sig2_matched[:midpoint]
    sig2b = sig2_matched[midpoint:]

    align_and_plot(sig1a, sig2a)
    align_and_plot(sig1b, sig2b)

def load_and_align(file_desc):
    sig1, sig2 = load_two_wav_files(file_desc)

    align_and_plot(sig1, sig2)


def main():
    filedesc1 = "20260609_173250"
    filedesc2 = "20260609_180357"
    load_and_align(filedesc2)

if __name__ == "__main__":
    main()