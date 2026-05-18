import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from recording import get_two_signals, FS

def find_claps(sig, height=0.2, min_dist_sec=0.5):
    """Uses scipy's robust peak finding."""
    # We look at absolute values because polarity might be flipped
    peaks, properties = signal.find_peaks(np.abs(sig), height=height, distance=int(min_dist_sec * FS))
    return peaks, properties['peak_heights']

def calculate_lag_restricted(sig1, sig2, idx1, window_size_sec=0.3, search_range_sec=1.0, center_in_2=None):
    """
    Correlates a snippet of sig1 against a RESTRICTED area of sig2.
    This is the key to preventing 'clap confusion'.
    """
    half_win = int((window_size_sec / 2) * FS)
    s1_start = max(0, idx1 - half_win)
    s1_end = min(len(sig1), idx1 + half_win)
    s1_snippet = sig1[s1_start:s1_end]
    
    if center_in_2 is None:
        center_in_2 = idx1
        
    search_win = int((search_range_sec / 2) * FS)
    s2_start = max(0, int(center_in_2 - search_win))
    s2_end = min(len(sig2), int(center_in_2 + search_win))
    s2_segment = sig2[s2_start:s2_end]
    
    if len(s1_snippet) == 0 or len(s2_segment) == 0:
        return 0
        
    # Cross-correlate the snippet against the restricted segment
    correlation = signal.correlate(s2_segment, s1_snippet, mode='full', method='fft')
    lags = signal.correlation_lags(len(s2_segment), len(s1_snippet), mode='full')
    
    best_idx = np.argmax(np.abs(correlation))
    
    # Convert local segment lag back to global signal lag
    # Result = (index in s2_segment) + (s2_segment_start_offset) - (index in s1)
    actual_lag = (lags[best_idx] + s2_start) - s1_start
    
    return actual_lag

def main():
    duration = 10
    print(f"--- Robust Clock Drift Experiment ---")
    print(f"Recording {duration}s. Use multiple sharp claps.")
    
    sig1, sig2 = get_two_signals(duration=duration)
    
    # 1. Find all claps in Mic 1 with a slightly adaptive threshold
    # We take 40% of the loudest peak as the minimum height for others
    initial_peaks, initial_heights = find_claps(sig1, height=0.1)
    if len(initial_peaks) == 0:
        print("No claps detected. Try clapping louder or lowering the height threshold.")
        return
        
    max_h = np.max(initial_heights)
    claps1, heights1 = find_claps(sig1, height=max_h * 0.4) 
    print(f"Detected {len(claps1)} valid claps.")

    # 2. Find the GLOBAL lag using the loudest clap first
    loudest_idx = claps1[np.argmax(heights1)]
    global_lag = calculate_lag_restricted(sig1, sig2, loudest_idx, search_range_sec=2.0)
    print(f"Global Alignment Reference (loudest clap): {global_lag} samples")

    # 3. Calculate precise lags for EVERY clap, restricted to avoid confusion
    results = []
    for i, c_idx in enumerate(claps1):
        # We search only +/- 200ms around the expected position in Mic 2
        # Expected = c_idx + global_lag
        precise_lag = calculate_lag_restricted(sig1, sig2, c_idx, 
                                              search_range_sec=0.4, 
                                              center_in_2=c_idx + global_lag)
        results.append(precise_lag)
        print(f"  Clap {i+1} at {c_idx/FS:.2f}s: Precise Lag = {precise_lag} samples")

    # 4. Drift Analysis
    total_drift = 0
    time_between = 0
    drift_rate = 0
    if len(results) >= 2:
        total_drift = results[-1] - results[0]
        time_between = (claps1[-1] - claps1[0]) / FS
        drift_rate = total_drift / time_between
        print(f"\n--- DRIFT RESULTS ---")
        print(f"Drift over {time_between:.2f}s: {total_drift} samples")
        print(f"Drift Rate: {drift_rate:.4f} samples/sec")
        print(f"Clock Error: { (drift_rate / FS) * 1e6 :.2f} parts-per-million (ppm)")

    # 5. Robust Visualization
    t = np.arange(len(sig1)) / FS
    
    # We'll determine the number of subplots dynamically
    num_plots = 1
    if len(results) >= 2:
        num_plots = 3
        
    plt.figure(figsize=(12, 4 * num_plots))
    
    # Top: Aligned Zoom (First Clap)
    plt.subplot(num_plots, 1, 1)
    plt.plot(t, sig1, label="Mic 1", color='blue')
    # Shift Mic 2 by the FIRST clap's lag for reference
    plt.plot(t - results[0]/FS, sig2, label="Mic 2 (shifted)", color='red', alpha=0.7)
    plt.title(f"Alignment at First Clap (Reference)")
    plt.xlim(claps1[0]/FS - 0.05, claps1[0]/FS + 0.05)
    plt.legend()
    plt.grid(True)

    if len(results) >= 2:
        # Middle: Aligned Zoom (Last Clap) - Showing the Drift
        plt.subplot(num_plots, 1, 2)
        plt.plot(t, sig1, label="Mic 1", color='blue')
        plt.plot(t - results[0]/FS, sig2, label="Mic 2 (shifted by FIRST lag)", color='red', alpha=0.7)
        plt.title(f"Looking at Last Clap (The gap shows the {total_drift} sample drift)")
        plt.xlim(claps1[-1]/FS - 0.05, claps1[-1]/FS + 0.05)
        plt.legend()
        plt.grid(True)

        # Bottom: Drift Plot
        plt.subplot(num_plots, 1, 3)
        plt.plot(claps1 / FS, results, 'o-', markersize=8)
        plt.title("Measured Lag at Each Clap (Should be a straight line)")
        plt.xlabel("Time of Clap (seconds)")
        plt.ylabel("Lag (samples)")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
