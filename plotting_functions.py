import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from archive.recording import FS

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

def get_top_n_peak_times(signal_data, n_points, fs, min_distance_sec=0.5):
    """
    Scans the audio to find the actual timestamps of the loudest distinct events.
    """
    abs_sig = np.abs(signal_data)

    # Require peaks to be separated by at least min_distance_sec to avoid echoes
    distance_samples = int(min_distance_sec * fs)
    peaks, _ = find_peaks(abs_sig, distance=distance_samples)

    if len(peaks) == 0:
        return []

    # Sort the found peaks by how loud they are (descending order)
    peak_amplitudes = abs_sig[peaks]
    sorted_peak_indices = np.argsort(peak_amplitudes)[::-1]

    # Grab the top N loudest peaks
    top_n_peaks = peaks[sorted_peak_indices[:n_points]]

    # Sort them back chronologically so the plots flow logically in time
    top_n_peaks = np.sort(top_n_peaks)

    # Convert the sample indices to time in seconds
    return top_n_peaks / fs

def compare_two_signals_at_multiple_points(sig1, sig2, n_points=6, fs=FS, zoom_window=0.1):
    # Find the actual claps instead of random times
    point_arr = get_top_n_peak_times(sig1, n_points, fs)

    if len(point_arr) == 0:
        print("No peaks found to plot.")
        return

    # Adjust n_points just in case the audio had fewer than N claps total
    n_points = len(point_arr)
    time_len = len(sig1) / fs
    time_axis1 = np.arange(len(sig1)) / fs
    time_axis2 = np.arange(len(sig2)) / fs

    # sharey=True ensures all plots use the exact same amplitude scale
    fig, axes = plt.subplots(n_points, 1, figsize=(12, 2 * n_points), sharey=True, constrained_layout=True)

    if n_points == 1:
        axes = [axes]

    for ax, point_time in zip(axes, point_arr):
        x_min = max(0, point_time - zoom_window)
        x_max = min(time_len, point_time + zoom_window)

        ax.plot(time_axis1, sig1, label="Mic 1", color='blue', alpha=0.7)
        ax.plot(time_axis2, sig2, label="Mic 2", color='red', alpha=0.7)

        ax.set_title(f"Loudest Event at {point_time:.2f}s", fontsize=11)
        ax.set_xlim(x_min, x_max)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Only put the legend on the very first plot
        if ax == axes[0]:
            ax.legend(loc="upper right")

    fig.supxlabel("Time (seconds)")
    fig.supylabel("Amplitude")
    fig.suptitle(f"Microphone Comparison at Top {n_points} Loudest Events", fontsize=14, fontweight='bold')

    plt.show()


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

def compare_n_signals_at_multiple_points(signals, n_points=6, fs=FS, zoom_window=0.1):
    if not signals:
        return

    # Find peaks using the first mic as the reference
    point_arr = get_top_n_peak_times(signals[0], n_points, fs)

    if len(point_arr) == 0:
        print("No peaks found to plot.")
        return

    n_points = len(point_arr)
    time_len = len(signals[0]) / fs
    time_axes = [np.arange(len(sig)) / fs for sig in signals]

    fig, axes = plt.subplots(n_points, 1, figsize=(12, 2 * n_points), sharey=True, constrained_layout=True)

    if n_points == 1:
        axes = [axes]

    for ax, point_time in zip(axes, point_arr):
        x_min = max(0, point_time - zoom_window)
        x_max = min(time_len, point_time + zoom_window)

        for i, (sig, t_axis) in enumerate(zip(signals, time_axes)):
            ax.plot(t_axis, sig, label=f"Mic {i + 1}", alpha=0.7)

        ax.set_title(f"Loudest Event at {point_time:.2f}s", fontsize=11)
        ax.set_xlim(x_min, x_max)
        ax.grid(True, linestyle='--', alpha=0.6)

        if ax == axes[0]:
            ax.legend(loc="upper right")

    fig.supxlabel("Time (seconds)")
    fig.supylabel("Amplitude")
    fig.suptitle(f"N-Microphone Comparison at Top {n_points} Loudest Events", fontsize=14, fontweight='bold')

    plt.show()


def plot_before_after_comparison(signals_before, signals_after, fs=FS, zoom_window=0.02):
    """
    Creates a 2x2 subplot grid comparing N signals before and after alignment.
    Top row: Full signals. Bottom row: Zoomed-in signals around the reference peak.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    half_window = int((zoom_window * fs) / 2)

    # -----------------------------------------------------
    # 1. Before Alignment (Full View)
    # -----------------------------------------------------
    for i, sig in enumerate(signals_before):
        axes[0, 0].plot(np.arange(len(sig)) / fs, sig, label=f"Signal {i + 1}", alpha=0.8)
    axes[0, 0].set_title("Before Alignment (Full)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].legend(loc="upper right")

    # -----------------------------------------------------
    # 2. After Alignment (Full View)
    # -----------------------------------------------------
    for i, sig in enumerate(signals_after):
        axes[0, 1].plot(np.arange(len(sig)) / fs, sig, label=f"Signal {i + 1}", alpha=0.8)
    axes[0, 1].set_title("After Alignment (Full)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].legend(loc="upper right")

    # -----------------------------------------------------
    # 3. Before Alignment (Zoomed View)
    # -----------------------------------------------------
    # We center the zoom on the absolute maximum of the first signal
    peak_b = np.argmax(np.abs(signals_before[0]))
    start_b = max(0, peak_b - half_window)

    for i, sig in enumerate(signals_before):
        end_b = min(len(sig), peak_b + half_window)
        # Check if start_b is within the bounds of this specific signal
        if start_b < len(sig):
            time_axis = np.arange(start_b, end_b) / fs
            axes[1, 0].plot(time_axis, sig[start_b:end_b], label=f"Signal {i + 1}", alpha=0.8)

    axes[1, 0].set_title(f"Before Alignment (Zoomed ~{zoom_window}s)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].legend(loc="upper right")

    # -----------------------------------------------------
    # 4. After Alignment (Zoomed View)
    # -----------------------------------------------------
    # Center the zoom on the aligned peak of the first signal
    peak_a = np.argmax(np.abs(signals_after[0]))
    start_a = max(0, peak_a - half_window)

    for i, sig in enumerate(signals_after):
        end_a = min(len(sig), peak_a + half_window)
        if start_a < len(sig):
            time_axis = np.arange(start_a, end_a) / fs
            axes[1, 1].plot(time_axis, sig[start_a:end_a], label=f"Signal {i + 1}", alpha=0.8)

    axes[1, 1].set_title(f"After Alignment (Zoomed ~{zoom_window}s)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()