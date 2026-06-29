# pyrefly: ignore [missing-import]
import torch
import numpy as np
from locate.n_mics_intersection import calculate_n_mic_points
from sound_file_handling import load_n_wav_files

# ==========================================
# CONFIGURATION
# ==========================================
# List of batch folders to process (found in audio_recordings/YYYY-MM-DD/)
TARGET_FILES = [
    # '2026-06-18_23-12-54',
    '2026-06-18_23-15-19',
    '2026-06-18_23-18-39',
    # '2026-06-18_23-30-45',

]

# Physical microphone coordinates in meters [x, y]
MICS_DICT = {
    0: np.array([1.5, 0.0]),
    1: np.array([-1.5, 0.0]),
    2: np.array([0.0, 3.0])
}

# Set to True to see the cross-correlation graphs and hyperbola plots
PLOT_STEPS = True
# ==========================================

class GPUAcousticLocalizer:
    def __init__(self, fs=44100):
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

    def gcc_phat(self, sig1, sig2, return_cc=False, expected_delay=0, max_deviation=None):
        """Calculates TDOA entirely on the GPU.
        Returns positive value when sig2 is delayed relative to sig1,
        matching the convention of alignment.calculate_cross_correlation.
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
        
        # Find peaks
        from scipy.signal import find_peaks
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
            
            # Find all peaks within 10% of the maximum peak (in the valid range)
            valid_peaks = peaks[peak_vals > 0.90 * max_peak_val]
            
            # Choose the peak closest to the expected delay
            best_peak = valid_peaks[np.argmin(np.abs(valid_peaks - target_idx))]
            peak_index = best_peak
        
        delay_samples = peak_index - center_index
        # Negate to match scipy correlate convention: positive = sig2 delayed
        delay_samples = -delay_samples
        
        if return_cc:
            return delay_samples, cc_shifted_np, peak_index
        return delay_samples

    def svd_denoise(self, matrix, energy_threshold=0.95):
        """GPU-accelerated Singular Value Decomposition."""
        # Compute SVD on the device
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        
        # Calculate cumulative energy to find the threshold elbow
        energy = S**2
        total_energy = torch.sum(energy)
        cumulative_energy = torch.cumsum(energy, dim=0) / total_energy
        
        # Find how many singular values to keep
        k_keep = torch.argmax((cumulative_energy >= energy_threshold).int()).item() + 1
        
        # Zero out the noise
        S_clean = torch.zeros_like(S)
        S_clean[:k_keep] = S[:k_keep]
        
        # Reconstruct the matrix: U * diag(S) * Vh
        clean_matrix = U @ torch.diag(S_clean) @ Vh
        return clean_matrix

    def process_timeline(self, mic_1, mic_2, mic_3, calib_window=(0, 10), target_window=(20, 30), plot=False):
        """
        Executes the 3-stage continuous recording pipeline.
        mic_1, mic_2, mic_3: Raw numpy arrays or lists.
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
        offset_1_2 = self.gcc_phat(calib_1, calib_2)
        offset_1_3 = self.gcc_phat(calib_1, calib_3)
        offset_2_3 = self.gcc_phat(calib_2, calib_3)
        print(f"Hardware Sync Offsets -> Mic 1-2: {offset_1_2} | Mic 1-3: {offset_1_3} | Mic 2-3: {offset_2_3}")
        
        # --- STAGE 2: Target Isolation & Denoising ---
        target_start, target_end = int(target_window[0] * self.fs), int(target_window[1] * self.fs)
        
        # Stack the target windows into a single GPU matrix (3, num_samples)
        target_matrix = torch.stack([
            t_mic1[target_start:target_end],
            t_mic2[target_start:target_end],
            t_mic3[target_start:target_end]
        ])
        
        print("Running SVD Denoising on target window...")
        
        if plot:
            # We skip plotting the raw messy matrix and jump straight to the denoised version
            pass

        clean_target_matrix = self.svd_denoise(target_matrix)

        if plot:
            # Align the signals using the hardware calibration offsets so they can be plotted accurately
            # Positive offset means mic N is delayed relative to mic 1.
            lags = [0, offset_1_2, offset_1_3]
            min_lag = min(lags)
            start_indices = [lag - min_lag for lag in lags]
            
            aligned_target_signals = []
            for i in range(3):
                aligned_target_signals.append(clean_target_matrix[i][start_indices[i]:].cpu().numpy())
                
            # Truncate to min length so they can be plotted together
            min_len = min(len(sig) for sig in aligned_target_signals)
            aligned_target_signals = [sig[:min_len] for sig in aligned_target_signals]
            
            from plotting_functions import plot_n_signals, plot_n_signals_around_max
            plot_n_signals(aligned_target_signals, fs=self.fs, title="Target Signals After Hardware Calibration Alignment (Denoised)")
            plot_n_signals_around_max(aligned_target_signals, fs=self.fs, title="Zoomed Target Signals After Calibration Alignment (Denoised)", zoom_window=0.05)
        
        clean_1 = clean_target_matrix[0]
        clean_2 = clean_target_matrix[1]
        clean_3 = clean_target_matrix[2]
        
        if plot:
            import matplotlib.pyplot as plt
            # Plot the spectrograms of the signals being correlated
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            fig.suptitle('Spectrograms of Denoised Target Signals')
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
        
        # Calculate maximum possible acoustic delay for target window
        c = 343 # speed of sound
        mics = list(MICS_DICT.values())
        max_dist = 0
        for i in range(len(mics)):
            for j in range(i+1, len(mics)):
                max_dist = max(max_dist, np.linalg.norm(mics[i] - mics[j]))
        
        # 1.5 * max_dist / c * fs
        max_acoustic_delay = int(1.5 * max_dist / c * self.fs)
        
        # --- STAGE 3: Final TDOA Calculation ---
        print("Calculating True TDOA...")
        if plot:
            import matplotlib.pyplot as plt
            raw_tdoa_1_2, cc_1_2, p_1_2 = self.gcc_phat(clean_1, clean_2, return_cc=True, expected_delay=offset_1_2, max_deviation=max_acoustic_delay)
            raw_tdoa_1_3, cc_1_3, p_1_3 = self.gcc_phat(clean_1, clean_3, return_cc=True, expected_delay=offset_1_3, max_deviation=max_acoustic_delay)
            raw_tdoa_2_3, cc_2_3, p_2_3 = self.gcc_phat(clean_2, clean_3, return_cc=True, expected_delay=offset_2_3, max_deviation=max_acoustic_delay)
            
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            fig.suptitle(f'GCC-PHAT Cross-Correlation vs. True Acoustic Delay')
            
            def plot_cc_subplot(ax, cc_array, raw_tdoa, offset, title):
                # Calculate lags in samples: positive means sig2 is delayed relative to sig1
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
                
                # Match alignment.py styling: purple line, alpha 0.8
                ax.plot(true_lags_seconds[sort_idx], cc_array[sort_idx], color='purple', alpha=0.8)
                
                # Match alignment.py styling: red dashed line, linewidth 2, legend label
                ax.axvline(x=true_delay_seconds, color='red', linestyle='--', linewidth=2,
                           label=f'Calculated Delay: {true_delay_seconds:.5f}s')
                
                ax.set_title(title)
                ax.set_ylabel("Correlation Magnitude")
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend(loc="upper right")
                
                # Zoom in around 0 seconds
                max_ac_sec = max_acoustic_delay / self.fs
                ax.set_xlim(-max_ac_sec * 2, max_ac_sec * 2)

            plot_cc_subplot(axs[0], cc_1_2, raw_tdoa_1_2, offset_1_2, "Mic 1 vs Mic 2")
            plot_cc_subplot(axs[1], cc_1_3, raw_tdoa_1_3, offset_1_3, "Mic 1 vs Mic 3")
            plot_cc_subplot(axs[2], cc_2_3, raw_tdoa_2_3, offset_2_3, "Mic 2 vs Mic 3")
            
            axs[2].set_xlabel("Delay (seconds)")
            plt.tight_layout()
            
            plt.show()
        else:
            raw_tdoa_1_2 = self.gcc_phat(clean_1, clean_2, expected_delay=offset_1_2, max_deviation=max_acoustic_delay)
            raw_tdoa_1_3 = self.gcc_phat(clean_1, clean_3, expected_delay=offset_1_3, max_deviation=max_acoustic_delay)
            raw_tdoa_2_3 = self.gcc_phat(clean_2, clean_3, expected_delay=offset_2_3, max_deviation=max_acoustic_delay)
        
        # True Delay = Raw Target Delay - Hardware Calibration Offset
        true_samples_1_2 = raw_tdoa_1_2 - offset_1_2
        true_samples_1_3 = raw_tdoa_1_3 - offset_1_3
        true_samples_2_3 = raw_tdoa_2_3 - offset_2_3
        
        if plot:
            import matplotlib.pyplot as plt
            # Align the signals perfectly using the true acoustic delay
            lags_true = [0, true_samples_1_2, true_samples_1_3]
            min_lag_true = min(lags_true)
            start_indices_true = [lag - min_lag_true for lag in lags_true]
            
            fully_aligned_signals = []
            for i in range(3):
                fully_aligned_signals.append(aligned_target_signals[i][start_indices_true[i]:])
                
            min_len_true = min(len(sig) for sig in fully_aligned_signals)
            fully_aligned_signals = [sig[:min_len_true] for sig in fully_aligned_signals]
            
            # Show the user the perfectly aligned signals overlaid on top of each other
            from plotting_functions import plot_n_signals_around_max
            plot_n_signals_around_max(fully_aligned_signals, fs=self.fs, title="Signals Overlaid After Full Acoustic Alignment", zoom_window=0.05)

        true_sec_1_2 = true_samples_1_2 / self.fs
        true_sec_1_3 = true_samples_1_3 / self.fs
        true_sec_2_3 = true_samples_2_3 / self.fs
        
        return {
            "Mic 1 to Mic 2": {"samples": true_samples_1_2, "seconds": true_sec_1_2},
            "Mic 1 to Mic 3": {"samples": true_samples_1_3, "seconds": true_sec_1_3},
            "Mic 2 to Mic 3": {"samples": true_samples_2_3, "seconds": true_sec_2_3}
        }

# ==========================================
# Example Usage
# ==========================================
def run_localization_batch(target_files, mics_dict, plot_steps=False):
    """Runs the full acoustic localization pipeline over a list of batch recordings."""
    localizer = GPUAcousticLocalizer(fs=44100)
    
    for filedesc in target_files:
        print("\n" + "="*50)
        print(f"Loading real recordings for batch: {filedesc}...")
        print("="*50)
        
        try:
            signals = load_n_wav_files(filedesc, n=3)
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
            calib_window=(0, 10),    # Look for the hardware sync clap here
            target_window=(20, 29),  # Look for the target sound here
            plot=plot_steps
        )
        
        print("\n--- FINAL RESULTS ---")
        for pair, data in results.items():
            print(f"{pair}: {data['samples']} samples ({data['seconds']:.6f} s)")
            
        # 2. Extract measured TDOAs for the math solver
        t_1_to_2 = results["Mic 1 to Mic 2"]["seconds"]
        t_1_to_3 = results["Mic 1 to Mic 3"]["seconds"]
        
        # Construct absolute arrival times
        t_arrivals = { 
            0: 0.0, 
            1: t_1_to_2, 
            2: t_1_to_3 
        }
        
        # 3. Mathematically triangulate the 2D position
        print("\n--- LOCATING SOURCE ---")
        points = calculate_n_mic_points(mics_dict, t_arrivals, plot=plot_steps)
        
        if points:
            est = np.mean(points, axis=0)
            print(f"Estimated Source Location: x = {est[0]:.4f}, y = {est[1]:.4f}")

if __name__ == "__main__":
    run_localization_batch(TARGET_FILES, MICS_DICT, plot_steps=PLOT_STEPS)