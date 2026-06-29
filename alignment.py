from scipy.linalg import _special_matrices
import csv
import os
import time


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift
from scipy.signal import hilbert, correlate, correlation_lags, butter, filtfilt

from plotting_functions import plot_both_signals, plot_both_signals_around_max, compare_two_signals_at_multiple_points, \
    plot_n_signals, compare_n_signals_at_multiple_points, plot_n_signals_around_max, plot_before_after_comparison
from sound_file_handling import FS, load_two_wav_files


def butter_bandpass_filter(signals, lowcut=500, highcut=8000, fs=44100, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return [filtfilt(b, a, sig) for sig in signals]

def tkeo(sig):
    """Calculates the Teager-Kaiser Energy Operator of a signal."""
    energy = np.zeros_like(sig)
    # The formula is: E(n) = x(n)^2 - x(n-1)*x(n+1)
    energy[1:-1] = sig[1:-1]**2 - sig[:-2] * sig[2:]
    return energy

def calc_tkeo(signals):
    return [tkeo(sig) for sig in signals]
    
def calc_envelopes(signals):
    return [np.abs(hilbert(sig)) for sig in signals]

def svd_denoise_matrix(matrix, energy_threshold=0.95):
    """CPU-accelerated Singular Value Decomposition using NumPy."""
    # Compute SVD (full_matrices=False gives the "economy" size SVD)
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)

    # Calculate cumulative energy to find the threshold elbow
    energy = S ** 2
    total_energy = np.sum(energy)
    cumulative_energy = np.cumsum(energy) / total_energy

    # Find how many singular values to keep
    # np.argmax returns the first index where the condition is True
    k_keep = np.argmax(cumulative_energy >= energy_threshold) + 1

    # Zero out the noise
    S_clean = np.zeros_like(S)
    S_clean[:k_keep] = S[:k_keep]

    # Reconstruct the matrix: U * diag(S) * Vh
    clean_matrix = U @ np.diag(S_clean) @ Vh
    return clean_matrix


def svd_denoise_signals(signals):
    matrix = np.vstack(signals)

    # Run the denoising algorithm
    cleaned_matrix = svd_denoise_matrix(matrix)

    # Unpack the matrix back into a list of 1D arrays
    # We iterate over the rows (cleaned_matrix.shape[0] is the number of mics)
    signals = [cleaned_matrix[i, :] for i in range(cleaned_matrix.shape[0])]

    return signals

def normalize_signals(signals):
    return [sig / (np.max(np.abs(sig)) + 1e-15) for sig in signals]

def process_signals_for_correlation(signals, preprocessing_parameters, gcc_phat=False):
    if preprocessing_parameters is None:
        env = False
        bandpass = False
        normalize = False
    else:
        env = preprocessing_parameters['env']
        bandpass = preprocessing_parameters['bandpass']
        normalize = preprocessing_parameters['normalize']

    if bandpass:
        signals = butter_bandpass_filter(signals)
        
    if env and not gcc_phat:
        signals = calc_envelopes(signals)
        
    if normalize:
        signals = normalize_signals(signals)

    return signals


def _calculate_regular_cross_correlation(sig1, sig2, fs=FS, verbose=False, return_cc=False):
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
        print("Regular cross correlation results:")
        print(f"Delay in samples: {lag_in_samples}, Delay in seconds: {delay_in_seconds:.5f}s")

        # 5. Interpret the results
        if lag_in_samples > 0:
            print("Result: Mic 1 heard the clap first.")
        elif lag_in_samples < 0:
            print("Result: Mic 2 heard the clap first.")
        else:
            print("Result: Incredible! The microphones are perfectly aligned.")
        print("-" * 30)


        # # We plot the absolute value of the correlation to clearly see the magnitude peak
        # lags_in_seconds = lags / fs
        # plt.figure(figsize=(10, 4))
        # plt.plot(lags_in_seconds, np.abs(correlation), color='purple', alpha=0.8)
        #
        # # Draw a vertical dashed red line exactly where the maximum peak occurs
        # plt.axvline(x=delay_in_seconds, color='red', linestyle='--', linewidth=2,
        #             label=f'Calculated Delay: {delay_in_seconds:.5f}s')
        #
        # plt.title("Cross-Correlation vs. Delay Time")
        # plt.xlabel("Delay (seconds)")
        # plt.ylabel("Correlation Magnitude")
        # plt.grid(True, linestyle='--', alpha=0.6)
        # plt.legend(loc="upper right")
        # plt.tight_layout()
        # plt.show()

    save_stats(sig1, sig2, lag_in_samples)

    if return_cc:
        return lag_in_samples, correlation, lags
    return lag_in_samples

def _calculate_gcc_phat(sig1, sig2, fs=FS, verbose=False, return_cc=False):

    n = len(sig1) + len(sig2) - 1
    n_fft = 1 << (n - 1).bit_length()

    SIG1 = fft(sig1, n=n_fft)
    SIG2 = fft(sig2, n=n_fft)

    # Calculate standard cross-correlation (sig2 correlated with sig1) to match scipy convention
    cross_power = SIG2 * np.conj(SIG1)
    cross_power_mag = np.abs(cross_power)
    
    # 0.01 to 0.05 is a standard sweet spot for noisy environments
    threshold = 0.01 * np.max(cross_power_mag) 
    
    # Use the threshold to regularize the phase transform denominator
    phat_weight = cross_power / (cross_power_mag + threshold)

    cc = np.real(fftshift(ifft(phat_weight)))

    # Find peak
    lag_in_samples = np.argmax(cc) - (n_fft // 2)

    if verbose:

        print("-" * 30)
        print("GCC-PHAT results:")
        delay_in_seconds = lag_in_samples / fs
        print(f"Delay in samples: {lag_in_samples}, Delay in seconds: {delay_in_seconds:.5f}s")

        # 5. Interpret the results
        if lag_in_samples > 0:
            print("Result: Mic 1 heard the clap first.")
        elif lag_in_samples < 0:
            print("Result: Mic 2 heard the clap first.")
        else:
            print("Result: Incredible! The microphones are perfectly aligned.")
        print("-" * 30)
        # # --- PLOTTING CODE ---
        # # Generate the x-axis (lags in seconds)
        # lags = np.arange(-n_fft // 2, n_fft // 2)
        # lags_in_seconds = lags / fs
        #
        # plt.figure(figsize=(10, 4))
        # # Plot the raw GCC-PHAT correlation data
        # plt.plot(lags_in_seconds, cc, color='teal', alpha=0.8)
        #
        # # Draw a vertical dashed red line at the peak
        # plt.axvline(x=delay_in_seconds, color='red', linestyle='--', linewidth=2,
        #             label=f'Peak Delay: {delay_in_seconds:.5f}s', alpha=0.5)
        #
        # plt.title("GCC-PHAT Correlation Magnitude vs. Delay")
        # plt.xlabel("Delay (seconds)")
        # plt.ylabel("Correlation Magnitude")
        # plt.grid(True, linestyle='--', alpha=0.6)
        # plt.legend(loc="upper right")
        # plt.tight_layout()
        # plt.show()

    if return_cc:
        lags = np.arange(-n_fft // 2, n_fft // 2)
        return lag_in_samples, cc, lags
    return lag_in_samples

def _calculate_regular_then_gcc_phat(sig1, sig2, fs=FS, verbose=False, env=False, return_cc=False):
    if verbose:
        print("~" * 30)
        print("Calculating regular cross correlation then GCC-PHAT:")

    if env:
        env1, env2 = calc_envelopes([sig1, sig2])
        coarse_lag = _calculate_regular_cross_correlation(env1, env2, fs=fs, verbose=verbose)
    else:
        coarse_lag = _calculate_regular_cross_correlation(sig1, sig2, fs=fs, verbose=verbose)

    aligned_sig1, aligned_sig2 = align_signals_given_lag(sig1, sig2, coarse_lag)

    sig1_crop, sig2_crop = _crop_two_aligned_signals_around_max(aligned_sig1, aligned_sig2, fs=fs)

    if env:
        sig1_crop = np.abs(hilbert(sig1_crop))
        sig2_crop = np.abs(hilbert(sig2_crop))

    hann_window = np.hanning(len(sig1_crop))
    sig1_crop = sig1_crop * hann_window
    sig2_crop = sig2_crop * hann_window

    if return_cc:
        fine_lag, cc, lags = _calculate_gcc_phat(sig1_crop, sig2_crop, verbose=verbose, return_cc=True)
        # Shift lags by coarse_lag so that the returned lags correspond to original frames
        lags_shifted = lags + coarse_lag
        return coarse_lag + fine_lag, cc, lags_shifted
    else:
        fine_lag = _calculate_gcc_phat(sig1_crop, sig2_crop, verbose=verbose)

    if verbose:
        print("~" * 30)

    return coarse_lag + fine_lag

def _crop_two_aligned_signals_around_max(sig1_aligned, sig2_aligned, fs=FS, window_sec=0.005):
    # Now that the claps overlap, find the absolute loudest peak
    peak_idx = np.argmax(np.abs(sig1_aligned))

    # Crop a very tight window (e.g., 0.05s) around that single peak
    half_window = int(window_sec * fs)
    start_idx = max(0, peak_idx - half_window)
    end_idx = min(len(sig1_aligned), peak_idx + half_window)

    sig1_crop = sig1_aligned[start_idx:end_idx]
    sig2_crop = sig2_aligned[start_idx:end_idx]

    return sig1_crop, sig2_crop

def calculate_cross_correlation(sig1 ,sig2, preprocessing_parameters, fs=FS, verbose=False, gcc_phat=False, env=False, bandpass=False, normalize=False, svd=False, return_cc=False):
    
    sig1, sig2 = process_signals_for_correlation([sig1, sig2], preprocessing_parameters, gcc_phat)
    if gcc_phat:
        return _calculate_gcc_phat(sig1, sig2, fs=fs, verbose=verbose, return_cc=return_cc)
        #return _calculate_regular_then_gcc_phat(sig1, sig2, fs=fs, verbose=verbose, env=env, return_cc=return_cc)

    return _calculate_regular_cross_correlation(sig1, sig2, fs=fs, verbose=verbose, return_cc=return_cc)



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

def align_two_signals(sig1, sig2, preprocessing_parameters, plot=True, verbose=False, gcc_phat=False):
    lag_in_samples = calculate_cross_correlation(sig1, sig2, preprocessing_parameters=preprocessing_parameters, verbose=verbose, gcc_phat=gcc_phat)

    aligned1, aligned2 = align_signals_given_lag(sig1, sig2, lag_in_samples)

    if plot:
        signals = [sig1, sig2]
        aligned = [aligned1, aligned2]
        plot_before_after_comparison(signals, aligned)


    return aligned1, aligned2, lag_in_samples

def calculate_lags_and_align_n_signals(signals, preprocessing_parameters, fs=FS, verbose=False, gcc_phat=False):
    """
    Aligns a list of N signals using signals[0] as the reference.
    """
    n = len(signals)
    lags = [0] * n  # Reference has 0 lag to itself

    # 1. Calculate lags relative to the first signal
    for i in range(1, n):
        print(f"CALCULATING LAG OF MIC {i}:")
        # We set verbose=False so it calculates quietly in the background
        lags[i] = calculate_cross_correlation(signals[0], signals[i], fs=fs, verbose=verbose, gcc_phat=gcc_phat, preprocessing_parameters=preprocessing_parameters)

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

def align_n_signals(signals, preprocessing_parameters, plot=True, verbose=False, gcc_phat=False):

    aligned_signals, lags = calculate_lags_and_align_n_signals(signals, preprocessing_parameters=preprocessing_parameters, verbose=verbose, gcc_phat=gcc_phat)

    if plot:
        plot_before_after_comparison(signals, aligned_signals)


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
    pass

if __name__ == "__main__":
    main()