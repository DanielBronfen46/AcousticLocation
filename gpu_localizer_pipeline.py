"""
gpu_localizer_pipeline.py

A standardized pipeline for acoustic localization utilizing GPU acceleration.
"""

import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

# pyrefly: ignore [missing-import]
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from locate.n_mics_intersection import calculate_n_mic_points, plot_tdoa_hyperbolas
from locate.three_way_intersection import calculate_points
from plotting_functions import plot_n_signals, plot_n_signals_around_max
from sound_file_handling import load_n_wav_files


# ==========================================
# CONFIGURATION
# ==========================================

@dataclass
class LocalizerConfig:
    """Configuration parameters for the GPU Acoustic Localizer."""
    # List of batch folders to process (found in audio_recordings/YYYY-MM-DD/)
    target_files: Tuple[str, ...] = (
        '2026-06-18_23-12-54',
        '2026-06-18_23-15-19',
        '2026-06-18_23-18-39'
    )
    
    # Physical microphone coordinates in meters [x, y]
    mics_dict: Dict[int, np.ndarray] = None
    
    # Set to True to see the cross-correlation graphs and hyperbola plots
    plot_steps: bool = True
    
    # Audio sampling rate
    fs: int = 44100
    
    # Toggle the center-favoring logic in GCC-PHAT peak finding
    favor_center: bool = True

    isolate_target: bool = True
    
    def __post_init__(self):
        if self.mics_dict is None:
            self.mics_dict = {
                0: np.array([1.5, 0.0]),
                1: np.array([-1.5, 0.0]),
                2: np.array([0.0, 3.0])
            }

# ==========================================
# CORE LOCALIZER
# ==========================================

class GPUAcousticLocalizer:
    """
    Handles acoustic localization signal processing with GPU acceleration.
    """
    
    def __init__(self, fs: int = 44100) -> None:
        """Initializes the localizer and selects the best hardware device."""
        self.fs = fs
        self.epsilon = 1e-15
        
        # Automatically select the best available hardware accelerator
        if torch.cuda.is_available():
            self.device = torch.device("cuda") # NVIDIA GPUs
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Apple Silicon (M1/M2/M3)
        else:
            self.device = torch.device("cpu")  # Fallback
            
        print(f"Initialized Localizer on device: {self.device}")



    def gcc_phat(self, sig1: torch.Tensor, sig2: torch.Tensor, return_cc: bool = False, expected_delay: int = 0,
                 max_deviation: Optional[int] = None, favor_center: bool = True, max_phys_delay: Optional[int] = None) -> Any:
        """
        Calculates Generalized Cross-Correlation Phase Transform (GCC-PHAT) on the GPU.
        
        Args:
            sig1: First signal tensor.
            sig2: Second signal tensor.
            return_cc: If True, returns (delay_samples, cc_array, peak_index).
            expected_delay: Center index expected shift.
            max_deviation: Maximum allowed peak shift from the expected delay.
            favor_center: Toggle fallback to center peak.
            max_phys_delay: Maximum physically possible acoustic delay in samples.
            
        Returns:
            delay_samples (int) or tuple (delay_samples, cc_array, peak_index).
            Positive delay_samples means sig2 is delayed relative to sig1.
        """
        n = sig1.shape[-1] + sig2.shape[-1] - 1
        n_padded = 2**(int(np.log2(n)) + 1)
        
        # Fast Fourier Transform on GPU
        X1 = torch.fft.rfft(sig1, n=n_padded)
        X2 = torch.fft.rfft(sig2, n=n_padded)
        
        cross_power = X1 * torch.conj(X2)
        phat_weight = cross_power / (torch.abs(cross_power) + self.epsilon)
        
        # Inverse FFT and shift
        cc = torch.fft.irfft(phat_weight, n=n_padded)
        cc_shifted = torch.fft.fftshift(cc)
        
        # Convert to numpy for peak finding
        cc_shifted_np = cc_shifted.cpu().numpy()
        peaks, _ = find_peaks(cc_shifted_np)
        
        center_index = n_padded // 2
        
        if len(peaks) == 0:
            peak_index = np.argmax(cc_shifted_np)
        else:
            target_idx = center_index - expected_delay
            
            if max_deviation is not None:
                # Filter peaks to only be within max_deviation of target_idx
                valid_mask = np.abs(peaks - target_idx) <= max_deviation
                if np.any(valid_mask):
                    peaks = peaks[valid_mask]
                else:
                    # If no peaks in range, simply find the max value in the allowed window
                    start_idx = max(0, target_idx - max_deviation)
                    end_idx = min(len(cc_shifted_np), target_idx + max_deviation + 1)
                    if start_idx < end_idx:
                        peaks = np.array([start_idx + np.argmax(cc_shifted_np[start_idx:end_idx])])

            peak_vals = cc_shifted_np[peaks]
            max_peak_val = np.max(peak_vals)
            
            absolute_max_peak = peaks[np.argmax(peak_vals)]
            
            if favor_center and max_phys_delay is not None:
                dist_from_zero_samples = np.abs(absolute_max_peak - target_idx)
                
                # If the max peak is physically possible (with small epsilon of 10 samples)
                if dist_from_zero_samples <= max_phys_delay + 10:
                    peak_index = absolute_max_peak
                else:
                    # It's an impossible echo, fall back to a smaller peak closer to 0
                    valid_peaks = peaks[peak_vals > 0.90 * max_peak_val]
                    best_peak = valid_peaks[np.argmin(np.abs(valid_peaks - target_idx))]
                    peak_index = best_peak
            else:
                peak_index = absolute_max_peak
        
        delay_samples = peak_index - center_index
        
        # Negate to match scipy correlate convention: positive = sig2 delayed
        delay_samples = -delay_samples
        
        if return_cc:
            return delay_samples, cc_shifted_np, peak_index, peaks
        return delay_samples


    def _pick_peak(self, cc_np: np.ndarray, peak_index: int, center_index: int) -> int:
        """Converts a raw peak index in the cc array back to a delay in samples."""
        return -(peak_index - center_index)

    def _isolate_target_event(self, matrix: torch.Tensor, window_sec: float = 1.0) -> torch.Tensor:
        """Finds the loudest acoustic event and slices a tight window around it."""
        window_samples = int(window_sec * self.fs)
        
        # Calculate total energy across all mics
        energy = torch.sum(matrix**2, dim=0)
        
        # Find the center of the loudest event
        peak_idx = torch.argmax(energy).item()
        
        # Create a tight boundary
        half_window = window_samples // 2
        start_idx = max(0, peak_idx - half_window)
        end_idx = min(matrix.shape[1], peak_idx + half_window)
        
        return matrix[:, start_idx:end_idx]
    # ==========================================
    # PLOTTING HELPERS
    # ==========================================
    
    def _plot_spectrograms(self, clean_1: torch.Tensor, clean_2: torch.Tensor, clean_3: torch.Tensor, title: str = 'Spectrograms of Denoised Target Signals') -> None:
        """Plots the frequency spectrograms of three denoised signals."""
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(title)
        
        axs[0].specgram(clean_1.cpu().numpy(), Fs=self.fs, NFFT=1024, noverlap=512, cmap='magma')
        axs[0].set_ylabel('Frequency (Hz)')
        axs[0].set_title('Mic 1')
        
        axs[1].specgram(clean_2.cpu().numpy(), Fs=self.fs, NFFT=1024, noverlap=512, cmap='magma')
        axs[1].set_ylabel('Frequency (Hz)')
        axs[1].set_title('Mic 2')
        
        axs[2].specgram(clean_3.cpu().numpy(), Fs=self.fs, NFFT=1024, noverlap=512, cmap='magma')
        axs[2].set_ylabel('Frequency (Hz)')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_title('Mic 3')
        
        plt.tight_layout()
        plt.show()

    def _plot_gcc_phat_correlations(self, cc_data: List[Tuple[np.ndarray, int, int]], max_acoustic_delay: Optional[int] = None, title: str = 'GCC-PHAT Cross-Correlation vs. True Acoustic Delay') -> None:
        """
        Plots the GCC-PHAT cross-correlation magnitudes for each mic pair.
        cc_data: List of tuples (cc_array, raw_tdoa, offset)
        """
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(title)
        
        titles = ["Mic 1 vs Mic 2", "Mic 1 vs Mic 3", "Mic 2 vs Mic 3"]
        
        for i, (cc_array, raw_tdoa, offset) in enumerate(cc_data):
            ax = axs[i]
            n_padded = len(cc_array)
            center = n_padded // 2
            lags_samples = center - np.arange(n_padded)
            
            # Represent the TRUE acoustic delay by subtracting hardware offset
            true_lags_samples = lags_samples - offset
            true_lags_seconds = true_lags_samples / self.fs
            
            # True delay of the selected peak
            true_delay_samples = raw_tdoa - offset
            true_delay_seconds = true_delay_samples / self.fs
            
            # Sort arrays so matplotlib plots strictly left-to-right
            sort_idx = np.argsort(true_lags_seconds)
            
            ax.plot(true_lags_seconds[sort_idx], cc_array[sort_idx], color='purple', alpha=0.8)
            ax.axvline(x=true_delay_seconds, color='red', linestyle='--', linewidth=2,
                       label=f'Calculated Delay: {true_delay_seconds:.5f}s')
            
            ax.set_title(titles[i])
            ax.set_ylabel("Correlation Magnitude")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc="upper right")
            
            # Zoom in if max_acoustic_delay is provided
            if max_acoustic_delay is not None:
                max_ac_sec = max_acoustic_delay / self.fs
                ax.set_xlim(-max_ac_sec * 2, max_ac_sec * 2)

        axs[2].set_xlabel("Delay (seconds)")
        plt.tight_layout()
        plt.show()

    def _plot_aligned_signals(self, clean_matrix: torch.Tensor, lags: List[int], title: str) -> None:
        """Aligns the target matrix signals by given delays and plots them overlaid."""
        min_lag = min(lags)
        start_indices = [lag - min_lag for lag in lags]
        
        aligned_signals = []
        for i in range(3):
            aligned_signals.append(clean_matrix[i][start_indices[i]:].cpu().numpy())
            
        min_len = min(len(sig) for sig in aligned_signals)
        aligned_signals = [sig[:min_len] for sig in aligned_signals]
        
        plot_n_signals_around_max(aligned_signals, fs=self.fs, title=title, zoom_window=0.05)

    def _plot_aligned_spectrograms(self, clean_matrix: torch.Tensor, lags: List[int], title: str) -> None:
        """Aligns the signals by given delays and plots their spectrograms stacked around the peak."""
        min_lag = min(lags)
        start_indices = [lag - min_lag for lag in lags]
        
        aligned_signals = []
        for i in range(3):
            aligned_signals.append(clean_matrix[i][start_indices[i]:].cpu().numpy())
            
        min_len = min(len(sig) for sig in aligned_signals)
        aligned_signals = [sig[:min_len] for sig in aligned_signals]
        
        zoom_window = 1.0  # 1 second before and after = 2 second snip
        peak_index = np.argmax(np.abs(aligned_signals[0]))
        half_window = int(zoom_window * self.fs)
        
        start_idx = max(0, peak_index - half_window)
        end_idx = min(min_len, peak_index + half_window)
        
        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(title + ' (2-Second Snippet)')
        
        NFFT = 1024
        noverlap = 768
        
        for i in range(3):
            sig_cropped = aligned_signals[i][start_idx:end_idx]
            axs[i].specgram(sig_cropped, Fs=self.fs, NFFT=NFFT, noverlap=noverlap, cmap='magma')
            axs[i].set_ylabel('Freq (Hz)')
            axs[i].set_title(f'Mic {i+1}')
            
        axs[2].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()


    # ==========================================
    # PIPELINE EXECUTION
    # ==========================================

    def process_timeline(self, mic_1: np.ndarray, mic_2: np.ndarray, mic_3: np.ndarray, 
                         mics_dict: Dict[int, np.ndarray],
                         calib_window: Tuple[int, int] = (0, 10), 
                         target_window: Tuple[int, int] = (20, 30), 
                         plot: bool = False,
                         favor_center: bool = True,
                         isolate_target: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Executes the 3-stage continuous recording pipeline.
        
        Args:
            mic_1, mic_2, mic_3: Raw numpy arrays for each microphone channel.
            mics_dict: Dictionary mapping mic index to physical coordinates.
            calib_window: Time window in seconds containing the hardware sync clap.
            target_window: Time window in seconds containing the target sound event.
            plot: If True, visualizes the signal processing steps.
            
        Returns:
            Dictionary containing the true TDOA samples and seconds for each pair.
        """
        print("\nStarting Timeline Processing...")
        
        # Move arrays to GPU memory
        t_mic1 = torch.tensor(mic_1, device=self.device, dtype=torch.float32)
        t_mic2 = torch.tensor(mic_2, device=self.device, dtype=torch.float32)
        t_mic3 = torch.tensor(mic_3, device=self.device, dtype=torch.float32)
        
        # --- STAGE 1: Calibration (Find Start Offsets) ---
        calib_start, calib_end = int(calib_window[0] * self.fs), int(calib_window[1] * self.fs)
        
        calib_1 = t_mic1[calib_start:calib_end]
        calib_2 = t_mic2[calib_start:calib_end]
        calib_3 = t_mic3[calib_start:calib_end]
        
        print("Calculating baseline sync offsets...")
        if plot:
            # 1. Signals of the part we are correlating at the start
            plot_n_signals_around_max(
                [calib_1.cpu().numpy(), calib_2.cpu().numpy(), calib_3.cpu().numpy()], 
                fs=self.fs, title="1. Calibration Window Signals (Raw)", zoom_window=0.05
            )
            
            offset_1_2, cc_calib_1_2, p_calib_1_2,_ = self.gcc_phat(calib_1, calib_2, return_cc=True)
            offset_1_3, cc_calib_1_3, p_calib_1_3,_ = self.gcc_phat(calib_1, calib_3, return_cc=True)
            offset_2_3, cc_calib_2_3, p_calib_2_3,_ = self.gcc_phat(calib_2, calib_3, return_cc=True)
            
            # 2. CC graphs of the whole period we are correlating at the start
            cc_data_calib = [
                (cc_calib_1_2, offset_1_2, 0),
                (cc_calib_1_3, offset_1_3, 0),
                (cc_calib_2_3, offset_2_3, 0)
            ]
            self._plot_gcc_phat_correlations(cc_data_calib, max_acoustic_delay=int(self.fs * 0.5), title="2. CC Graphs of Calibration")
            
            # 3. Signals of the part we are correlating at the start, after correction
            calib_matrix = torch.stack([calib_1, calib_2, calib_3])
            lags_calib = [0, offset_1_2, offset_1_3]
            self._plot_aligned_signals(calib_matrix, lags_calib, "3. Calibration Window Signals (After Correction)")
            
        else:
            offset_1_2 = self.gcc_phat(calib_1, calib_2)
            offset_1_3 = self.gcc_phat(calib_1, calib_3)
            offset_2_3 = self.gcc_phat(calib_2, calib_3)
            
        print(f"Hardware Sync Offsets -> Mic 1-2: {offset_1_2} | Mic 1-3: {offset_1_3} | Mic 2-3: {offset_2_3}")
        
        # --- STAGE 2: Target Isolation & Denoising ---
        target_start, target_end = int(target_window[0] * self.fs), int(target_window[1] * self.fs)
        
        target_matrix = torch.stack([
            t_mic1[target_start:target_end],
            t_mic2[target_start:target_end],
            t_mic3[target_start:target_end]
        ])
        
        full_1, full_2, full_3 = target_matrix[0], target_matrix[1], target_matrix[2]
        
        if isolate_target:
            print("Isolating target event for math calculation...")
            clean_target_matrix = self._isolate_target_event(target_matrix, window_sec=1.0)
            clean_1, clean_2, clean_3 = clean_target_matrix[0], clean_target_matrix[1], clean_target_matrix[2]
        else:
            print("Using full target event window for math calculation...")
            clean_1, clean_2, clean_3 = full_1, full_2, full_3
        
        if plot:
            # 4. The spectrogram of the part we are correlating at the end
            self._plot_spectrograms(full_1, full_2, full_3, title="4. Spectrogram of Target Event (Full Window)")
        
        # Calculate maximum possible acoustic delay for bounds checking
        c = 343 # speed of sound (m/s)
        mics = list(mics_dict.values())
        max_dist = max([np.linalg.norm(mics[i] - mics[j]) for i in range(len(mics)) for j in range(i+1, len(mics))])
        
        # Max physically possible acoustic delay in samples
        max_phys_delay = int(max_dist / c * self.fs)
        
        # Generous boundary for filtering peaks (1.5x physical distance)
        max_acoustic_delay = int(1.5 * max_dist / c * self.fs)
        
        # --- STAGE 3: Final TDOA Calculation ---
        print("Calculating True TDOA...")
        
        # Math calculation — always return_cc so we can retry alternative peaks on consistency failure
        raw_tdoa_1_2, cc_1_2, pidx_1_2, peaks_1_2 = self.gcc_phat(clean_1, clean_2, return_cc=True, expected_delay=offset_1_2, max_deviation=max_acoustic_delay, favor_center=favor_center, max_phys_delay=max_phys_delay)
        raw_tdoa_1_3, cc_1_3, pidx_1_3, peaks_1_3 = self.gcc_phat(clean_1, clean_3, return_cc=True, expected_delay=offset_1_3, max_deviation=max_acoustic_delay, favor_center=favor_center, max_phys_delay=max_phys_delay)
        raw_tdoa_2_3, cc_2_3, pidx_2_3, peaks_2_3 = self.gcc_phat(clean_2, clean_3, return_cc=True, expected_delay=offset_2_3, max_deviation=max_acoustic_delay, favor_center=favor_center, max_phys_delay=max_phys_delay)
        
        if plot:
            # Visualization uses the full 9-second clip
            _, cc_full_1_2, _, _ = self.gcc_phat(full_1, full_2, return_cc=True, expected_delay=offset_1_2, max_deviation=max_acoustic_delay, favor_center=favor_center, max_phys_delay=max_phys_delay)
            _, cc_full_1_3, _, _ = self.gcc_phat(full_1, full_3, return_cc=True, expected_delay=offset_1_3, max_deviation=max_acoustic_delay, favor_center=favor_center, max_phys_delay=max_phys_delay)
            _, cc_full_2_3, _, _ = self.gcc_phat(full_2, full_3, return_cc=True, expected_delay=offset_2_3, max_deviation=max_acoustic_delay, favor_center=favor_center, max_phys_delay=max_phys_delay)
            
            cc_data = [
                (cc_full_1_2, raw_tdoa_1_2, offset_1_2),
                (cc_full_1_3, raw_tdoa_1_3, offset_1_3),
                (cc_full_2_3, raw_tdoa_2_3, offset_2_3)
            ]
            # 5. The cc graphs of the whole period we are correlating at the end
            self._plot_gcc_phat_correlations(cc_data, max_acoustic_delay=None, title="5. CC Graphs of Target Event (Full Window)")
        
        # True Delay = Raw Target Delay - Hardware Calibration Offset
        true_samples_1_2 = raw_tdoa_1_2 - offset_1_2
        true_samples_1_3 = raw_tdoa_1_3 - offset_1_3
        true_samples_2_3 = raw_tdoa_2_3 - offset_2_3
        
        if plot:
            target_matrix = torch.stack([clean_1, clean_2, clean_3])
            lags_calib = [0, offset_1_2, offset_1_3]
            self._plot_aligned_signals(target_matrix, lags_calib, "6. Target Event Signals (After Calibration)")
            self._plot_aligned_spectrograms(target_matrix, lags_calib, "7. Target Event Spectrograms (After Calibration)")
        # --- TDOA Self-Consistency Check & Correction ---
        # By definition: t(1→3) = t(1→2) + t(2→3)
        # If one measurement is inconsistent with the other two, it likely picked a
        # spurious GCC-PHAT peak (e.g. a reflection). We detect and correct this.
        #
        # Threshold: max_phys_delay is the absolute upper bound for any single TDOA,
        # but the *self-consistency error* (t13 - t12 - t23) should be nearly zero —
        # only a few samples of GCC-PHAT measurement noise. We use max_phys_delay // 5
        # as a generous but tight-enough trigger to catch spurious peaks without
        # false-positiving on small noise.
        consistency_threshold = max(20, max_phys_delay // 5)
        
        expected_1_3 = true_samples_1_2 + true_samples_2_3
        expected_1_2 = true_samples_1_3 - true_samples_2_3
        expected_2_3 = true_samples_1_3 - true_samples_1_2

        errors = {
            "Mic 1 to Mic 3": abs(true_samples_1_3 - expected_1_3),
            "Mic 1 to Mic 2": abs(true_samples_1_2 - expected_1_2),
            "Mic 2 to Mic 3": abs(true_samples_2_3 - expected_2_3),
        }
        
        worst_pair = max(errors, key=errors.get)
        worst_error = errors[worst_pair]
        
        print(f"[CONSISTENCY] Self-consistency errors -> "
              f"1-2: {errors['Mic 1 to Mic 2']} | "
              f"1-3: {errors['Mic 1 to Mic 3']} | "
              f"2-3: {errors['Mic 2 to Mic 3']} samples  (threshold: {consistency_threshold})")
        
        discarded_pair = None
        if worst_error > consistency_threshold:
            # --- RETRY: try next-best peaks for the outlier pair before discarding ---
            outlier_info = {
                "Mic 1 to Mic 2": (cc_1_2, pidx_1_2, peaks_1_2, offset_1_2),
                "Mic 1 to Mic 3": (cc_1_3, pidx_1_3, peaks_1_3, offset_1_3),
                "Mic 2 to Mic 3": (cc_2_3, pidx_2_3, peaks_2_3, offset_2_3),
            }
            cc_out, pidx_out, peaks_out, offset_out = outlier_info[worst_pair]
            center_index = len(cc_out) // 2
            
            # Sort all alternative peaks by height (descending), skipping the one already tried
            alt_peaks = [p for p in peaks_out[np.argsort(cc_out[peaks_out])[::-1]] if p != pidx_out]
            
            retry_success = False
            for alt_peak in alt_peaks:
                candidate_raw = self._pick_peak(cc_out, alt_peak, center_index)
                candidate_true = candidate_raw - offset_out
                
                # Substitute candidate into the triplet and re-check consistency
                if worst_pair == "Mic 1 to Mic 2":
                    cand_1_2, cand_1_3, cand_2_3 = candidate_true, true_samples_1_3, true_samples_2_3
                elif worst_pair == "Mic 1 to Mic 3":
                    cand_1_2, cand_1_3, cand_2_3 = true_samples_1_2, candidate_true, true_samples_2_3
                else:
                    cand_1_2, cand_1_3, cand_2_3 = true_samples_1_2, true_samples_1_3, candidate_true
                
                retry_err = abs(cand_1_3 - (cand_1_2 + cand_2_3))
                if retry_err <= consistency_threshold:
                    print(f"[CONSISTENCY] '{worst_pair}' -- retrying with next-best peak: "
                          f"candidate={candidate_true} samples (error improved: {worst_error} -> {retry_err})")
                    # Accept the correction
                    if worst_pair == "Mic 1 to Mic 2":
                        true_samples_1_2 = candidate_true
                        raw_tdoa_1_2 = candidate_raw
                    elif worst_pair == "Mic 1 to Mic 3":
                        true_samples_1_3 = candidate_true
                        raw_tdoa_1_3 = candidate_raw
                    else:
                        true_samples_2_3 = candidate_true
                        raw_tdoa_2_3 = candidate_raw
                    retry_success = True
                    break
            
            if not retry_success:
                discarded_pair = worst_pair
                print(f"[CONSISTENCY] '{worst_pair}' -- no alternative peak worked. Discarding pair.")

        true_sec_1_2 = true_samples_1_2 / self.fs
        true_sec_1_3 = true_samples_1_3 / self.fs
        true_sec_2_3 = true_samples_2_3 / self.fs
        
        return {
            "Mic 1 to Mic 2": {"samples": true_samples_1_2, "seconds": true_sec_1_2},
            "Mic 1 to Mic 3": {"samples": true_samples_1_3, "seconds": true_sec_1_3},
            "Mic 2 to Mic 3": {"samples": true_samples_2_3, "seconds": true_sec_2_3},
            "worst_pair": discarded_pair
        }


# ==========================================
# BATCH EXECUTION
# ==========================================

def run_localization_batch(config: LocalizerConfig) -> None:
    """Runs the full acoustic localization pipeline over a list of batch recordings."""
    localizer = GPUAcousticLocalizer(fs=config.fs)
    
    for folder_name in config.target_files:
        print("\n" + "="*50)
        print(f"Loading real recordings for batch: {folder_name}...")
        print("="*50)
        
        try:
            # We assume your folder has files ending in 1.wav, 2.wav, 3.wav
            signals = load_n_wav_files(folder_name, n=3)
        except Exception as e:
            print(f"Failed to load files: {e}")
            continue

        if len(signals) < 3:
            print("Error: Could not load 3 microphone signals for this test.")
            continue
            
        mic_1, mic_2, mic_3 = signals[0], signals[1], signals[2]
        
        # 1. Run the acoustic signal processing
        results = localizer.process_timeline(
            mic_1, mic_2, mic_3, 
            mics_dict=config.mics_dict,
            calib_window=(0, 10),    # Look for the hardware sync clap here
            target_window=(30, 39),  # Look for the target sound here
            plot=config.plot_steps,
            favor_center=config.favor_center,
            isolate_target=config.isolate_target
        )
        
        print("\n--- FINAL RESULTS ---")
        worst_pair = results.get("worst_pair")
        for pair, data in results.items():
            if pair == "worst_pair": continue
            discarded_note = " [DISCARDED via consistency]" if pair == worst_pair else ""
            print(f"{pair}: {data['samples']} samples ({data['seconds']:.6f} s){discarded_note}")
            
        # 2. Extract relative arrival times for the math.
        # calculate_points / algebraic_intersection expects delta_AB = dA - dB (meters).
        # gcc_phat(sigA, sigB) returns negative when sigB is farther, so we negate
        # to get delta_AB = tA - tB in the correct sign convention.
        time_difs = {
            '01': -results["Mic 1 to Mic 2"]["seconds"],
            '02': -results["Mic 1 to Mic 3"]["seconds"],
            '12': -results["Mic 2 to Mic 3"]["seconds"]
        }
        
        print("\n--- LOCATING SOURCE ---")
        points = calculate_points(config.mics_dict, time_difs)
        
        # Select the intersection point computed purely from the two good pairs
        if worst_pair == "Mic 1 to Mic 2":
            valid_points = [points[2]] if points[2] is not None else []
        elif worst_pair == "Mic 1 to Mic 3":
            valid_points = [points[1]] if points[1] is not None else []
        elif worst_pair == "Mic 2 to Mic 3":
            valid_points = [points[0]] if points[0] is not None else []
        else:
            valid_points = [p for p in points if p is not None]
        
        if config.plot_steps:
            from locate.three_way_intersection import plot_three_way_hyperbolas
            est = np.mean(valid_points, axis=0) if valid_points else None
            plot_three_way_hyperbolas(config.mics_dict, time_difs, points=points, est_intersection=est)
        
        if valid_points:
            est = np.mean(valid_points, axis=0)
            print(f"Estimated Source Location: x = {est[0]:.4f}, y = {est[1]:.4f}")
        else:
            print("Failed to locate source (no valid intersections found).")


if __name__ == "__main__":
    import os
    
    date_dir = "audio_recordings/2026-06-28"
    recordings = tuple(
        d for d in sorted(os.listdir(date_dir)) 
        if os.path.isdir(os.path.join(date_dir, d))
    ) if os.path.exists(date_dir) else ()
    recordings = recordings[-1:]

    # Configuring mic positions
    CONFIG = LocalizerConfig(
        mics_dict={
            0: np.array([0.0, 0.0]),
            1: np.array([2.0, 0.0]),
            2: np.array([0.0, 2.2])
        },
        target_files=recordings,
        plot_steps=True,
        isolate_target=False
    )
    run_localization_batch(CONFIG)
