from scipy.interpolate import interp1d
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import butter, filtfilt, hilbert

from alignment import plot_both_signals, plot_both_signals_around_max, compare_two_signals_at_multiple_points
from recording import record_two_signals, FS, load_two_wav_signals, trim_zeroes
import matplotlib.pyplot as plt



def apply_lowpass_filter(sig, cutoff_freq, fs=FS, order=4):
    """
    Removes low-frequency rumble and DC baseline drift from a signal.
    Uses filtfilt to guarantee zero phase-shift (crucial for alignment).

    cutoff_freq: Frequencies below this (in Hz) will be heavily reduced.
                 40-50 Hz is standard to remove rumble without hurting voice or 200Hz tones.
    """
    # 1. Calculate the Nyquist frequency (half the sample rate)
    nyquist = 0.5 * fs

    # 2. Normalize the cutoff frequency for the digital filter
    normal_cutoff = cutoff_freq / nyquist

    # 3. Build a Butterworth high-pass filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # 4. Apply the filter forwards and backwards to preserve exact phase alignment
    filtered_sig = filtfilt(b, a, sig)

    return filtered_sig

def plot_hadamard_product(sig1, sig2, fs=FS, title="Element-wise Dot Product"):
    """
    Plots the element-wise product of two signals.
    Highly useful after alignment to see where the signals reinforce each other.
    """
    time_axis = np.arange(len(sig1)) / fs
    product = sig1 * sig2

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, product, color='green', alpha=0.8)

    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Product Amplitude")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_xy_signals(sig1, sig2, title="XY Plot of Signals", xlabel="Signal 1", ylabel="Signal 2", plot_style='-'):
    """
    Plots an XY graph of two signals against each other.
    Highly useful for checking phase alignment (Lissajous figures).
    """
    # Ensure both signals are exactly the same length before plotting
    min_length = min(len(sig1), len(sig2))
    x_data = sig1[:min_length]
    y_data = sig2[:min_length]

    # A square figure is usually best for XY phase plots
    plt.figure(figsize=(6, 6))

    # We use a slight transparency (alpha) because audio signals pack many points tightly
    plt.plot(x_data, y_data, plot_style, alpha=0.5, linewidth=1)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Draw zero-lines for reference
    plt.axhline(0, color='black', linewidth=0.8, alpha=0.7)
    plt.axvline(0, color='black', linewidth=0.8, alpha=0.7)

    plt.grid(True, linestyle='--', alpha=0.6)

    # Force axes to have equal scaling so phase shifts don't look skewed
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def find_sample_frequency_diff_IQ(sig1, sig2, f_played, fs=FS):
    time_axis = np.arange(len(sig1)) / fs

    # 2. In-Phase (I) Component (Your original logic)
    product_I = sig1 * sig2
    I = apply_lowpass_filter(product_I, 10.0, fs=fs)

    # 3. Quadrature (Q) Component (The 90-degree shifted logic)
    # scipy.signal.hilbert returns the analytic signal: real + j*imaginary
    # The imaginary part is the original signal shifted by -90 degrees
    sig1_shifted = np.imag(hilbert(sig1))
    product_Q = sig1_shifted * sig2
    Q = apply_lowpass_filter(product_Q, 10.0, fs=fs)

    # 4. Complex Envelope and Instantaneous Phase
    complex_envelope = I + 1j * Q

    # np.angle gets phase from -pi to pi.
    # np.unwrap stitches it into a continuous ascending/descending line
    instantaneous_phase = np.unwrap(np.angle(complex_envelope))

    # 5. Extract Frequency and Sign via Linear Regression
    # Phase equation: phase(t) = 2 * pi * f_fit * t + initial_phase
    # We fit a 1st-degree polynomial (a line) to get the slope
    slope, intercept = np.polyfit(time_axis, instantaneous_phase, 1)

    # f_fit will now be POSITIVE if sig1 is faster, NEGATIVE if sig2 is faster
    f_fit = slope / (2 * np.pi)

    # 6. Extract Start Delay (from the intercept)
    # The intercept is the initial phase difference in radians
    phi_fit = (intercept + np.pi) % (2 * np.pi) - np.pi
    phi_fit += np.pi / 2
    delta_t_ms = (phi_fit) / (2 * np.pi * f_played) * 1000



    # 7. Calculate Hardware Drift
    delta_fs = (f_fit * fs) / f_played

    # 8. Print Results
    print("--- IQ Demodulation Results ---")
    print(f"Beat Frequency : {f_fit:.6f} Hz")
    print(f"Initial Phase  : {phi_fit:.6f} rad")
    print("--- Hardware Clock Drift ---")
    print(f"f2 - f1        : {delta_fs:.6f} Hz")
    if delta_fs > 0:
        print("Diagnosis      : Mic 2 is ticking faster than Mic 1")
    else:
        print("Diagnosis      : Mic 1 is ticking faster than Mic 2")
    print("--- Sub-cycle Start Delay ---")
    print(f"Delay          : {delta_t_ms:.4f} ms")
    print("-------------------------------")

    # --- NEW: Plot the Fitted Sine Wave over the Dot Product ---
    # Reconstruct the fitted wave using Envelope Amplitude * cos(Fitted Phase)
    amplitude = np.abs(complex_envelope)
    fitted_phase = slope * time_axis + intercept
    fitted_wave = amplitude * np.cos(fitted_phase)

    plt.figure(figsize=(10, 4))
    # Plot raw dot product in background
    plt.plot(time_axis, product_I, color='green', alpha=0.3, label='Raw Dot Product')
    # Plot the filtered signal (I component)
    plt.plot(time_axis, I, color='blue', alpha=0.6, label='Filtered (I Component)')
    # Plot the perfectly fitted wave derived from the linear phase regression
    plt.plot(time_axis, fitted_wave, color='red', linestyle='--', linewidth=2, label='Fitted Sine Wave')

    plt.title("Dot Product vs. Fitted Demodulated Sine Wave")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Optional: Plot the phase line to verify it is straight
    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, instantaneous_phase, label='Unwrapped Phase')
    plt.plot(time_axis, slope * time_axis + intercept, 'r--', label='Linear Fit')
    plt.title("Instantaneous Phase over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (radians)")
    plt.legend()
    plt.grid(True)
    plt.show()

    delta_t_s = delta_t_ms / 1000
    return delta_fs, delta_t_s

def match_amplitude(target_sig, sig_to_scale, method='rms'):
    """
    Scales 'sig_to_scale' so its amplitude matches 'target_sig'.

    Available methods:
    - 'rms' (default): Matches the overall energy/loudness. Robust against spikes.
    - 'peak': Matches the absolute highest point. Good for strict visual bounding.
    - 'lstsq': Least-squares fit. Mathematically optimal for phase-aligned signals.
    """

    if method == 'rms':
        # 1. Root Mean Square (Energy Match)
        rms_target = np.sqrt(np.mean(target_sig ** 2))
        rms_source = np.sqrt(np.mean(sig_to_scale ** 2))

        if rms_source == 0:
            return sig_to_scale

        scale_factor = rms_target / rms_source

    elif method == 'peak':
        # 2. Absolute Peak Match
        peak_target = np.max(np.abs(target_sig))
        peak_source = np.max(np.abs(sig_to_scale))

        if peak_source == 0:
            return sig_to_scale

        scale_factor = peak_target / peak_source

    elif method == 'lstsq':
        # 3. Least Squares (Optimal mathematical fit)
        # Calculates 'a' to minimize the difference: sum((target - a * source)^2)
        # Note: Both signals MUST be exactly the same length for this method.
        min_len = min(len(target_sig), len(sig_to_scale))
        t_sig = target_sig[:min_len]
        s_sig = sig_to_scale[:min_len]

        scale_factor = np.dot(t_sig, s_sig) / np.dot(s_sig, s_sig)

    else:
        raise ValueError("Method must be 'rms', 'peak', or 'lstsq'")

    print(f"Scaling Mic 2 by a factor of: {scale_factor:.4f} (Method: {method})")

    # Apply the scaling
    scaled_sig = sig_to_scale * scale_factor

    return scaled_sig, scale_factor

def fix_frequency_and_phase(sig1, sig2, delta_fs, delta_t_s, fs=FS):
    """
    Aligns sig2 to sig1 by correcting both hardware clock drift (delta_fs)
    and sub-cycle start delay (delta_t_s).
    """

    # 2. Calculate the effective sampling rate of Mic 2
    # If delta_fs is positive, Mic 2 is ticking faster.
    fs2_effective = fs + delta_fs

    # 3. Build the TRUE physical timeline of sig2
    # The first sample (n=0) occurred at delta_t_sec.
    # Each subsequent sample occurred exactly (1 / fs2_effective) seconds later.
    t_true_sig2 = np.arange(len(sig2)) / fs2_effective + delta_t_s

    # 4. Build the TARGET timeline (Mic 1's timeline)
    # We want both signals evaluated exactly at these target timestamps.
    t_target = np.arange(len(sig1)) / fs

    # 5. Create the Interpolator
    # We use 'cubic' interpolation because it preserves audio curves smoothly.
    # bounds_error=False and fill_value=0.0 ensure that any empty space created
    # by shifting the array is filled with digital silence (zeros) instead of crashing.

    interpolator = interp1d(
        t_true_sig2,
        sig2,
        kind='cubic',
        bounds_error=False,
        fill_value=0.0
    )

    # 6. Generate the perfectly aligned sig2
    sig2_resampled = interpolator(t_target)

    sig1_aligned, sig2_aligned = trim_zeroes(sig1, sig2_resampled)

    return sig1_aligned, sig2_aligned


def calibrate_sine_waves_and_plot(sig1, sig2, f_played, fs=FS):
    compare_two_signals_at_multiple_points(sig1, sig2, n_points=4, fs=fs)
    #plot_xy_signals(sig1, sig2)

    # 1. Get the perfect mathematical ground truths (Now bug-free)
    delta_fs, delta_t_s = find_sample_frequency_diff_IQ(sig1, sig2, f_played, fs=fs)

    # 2. Fix the frequency and phase deterministically
    sig1_fixed, sig2_aligned = fix_frequency_and_phase(sig1, sig2, delta_fs, delta_t_s, fs=fs)

    sig2_fixed, scale_factor = match_amplitude(sig1_fixed, sig2_aligned, method='rms')

    # 3. Plot the final results
    compare_two_signals_at_multiple_points(sig1_fixed, sig2_fixed, n_points=4, fs=fs)
    #plot_xy_signals(sig1_fixed, sig2_fixed)
    plot_hadamard_product(sig1_fixed, sig2_fixed, fs=fs, title="Fixed Dot Product")

    return sig1_fixed, sig2_fixed, delta_fs, delta_t_s, scale_factor


def record_sine_signals_and_fix(f_played, duration):
    sig1, sig2 = record_two_signals(duration=duration)


    calibrate_sine_waves_and_plot(sig1, sig2, f_played)

def load_sine_signals_and_fix(f_played, file_desc):
    sig1, sig2 = load_two_wav_signals(file_desc)

    calibrate_sine_waves_and_plot(sig1, sig2, f_played)


def main():
    file_desc = "recording_2026-05-31_12-45-41_200hz_20s"
    #file_desc = "recording_2026-05-31_10-41-31_200hz_5s"
    record_sine_signals_and_fix(f_played=200, duration=40)
    #load_sine_signals_and_fix(f_played=200, file_desc=file_desc)

if __name__ == "__main__":
    main()