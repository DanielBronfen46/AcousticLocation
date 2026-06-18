import torch
import numpy as np

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

    def gcc_phat(self, sig1, sig2):
        """Calculates TDOA entirely on the GPU."""
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
        
        # Find peak
        center_index = n_padded // 2
        peak_index = torch.argmax(cc_shifted).item()
        
        delay_samples = peak_index - center_index
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

    def process_timeline(self, mic_1, mic_2, mic_3, calib_window=(0, 10), target_window=(20, 30)):
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
        print(f"Hardware Sync Offsets -> Mic 2: {offset_1_2} samples | Mic 3: {offset_1_3} samples")
        
        # --- STAGE 2: Target Isolation & Denoising ---
        target_start, target_end = int(target_window[0] * self.fs), int(target_window[1] * self.fs)
        
        # Stack the target windows into a single GPU matrix (3, num_samples)
        target_matrix = torch.stack([
            t_mic1[target_start:target_end],
            t_mic2[target_start:target_end],
            t_mic3[target_start:target_end]
        ])
        
        print("Running SVD Denoising on target window...")
        clean_target_matrix = self.svd_denoise(target_matrix)
        
        clean_1 = clean_target_matrix[0]
        clean_2 = clean_target_matrix[1]
        clean_3 = clean_target_matrix[2]
        
        # --- STAGE 3: Final TDOA Calculation ---
        print("Calculating True TDOA...")
        raw_tdoa_1_2 = self.gcc_phat(clean_1, clean_2)
        raw_tdoa_1_3 = self.gcc_phat(clean_1, clean_3)
        
        # True Delay = Raw Target Delay - Hardware Calibration Offset
        true_samples_1_2 = raw_tdoa_1_2 - offset_1_2
        true_samples_1_3 = raw_tdoa_1_3 - offset_1_3
        
        true_sec_1_2 = true_samples_1_2 / self.fs
        true_sec_1_3 = true_samples_1_3 / self.fs
        
        return {
            "Mic 1 to Mic 2": {"samples": true_samples_1_2, "seconds": true_sec_1_2},
            "Mic 1 to Mic 3": {"samples": true_samples_1_3, "seconds": true_sec_1_3}
        }

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    fs = 44100
    total_length = fs * 35 # 35 seconds of recording
    
    # Simulating the raw data
    mic_1 = np.random.normal(0, 0.1, total_length)
    mic_2 = np.random.normal(0, 0.1, total_length)
    mic_3 = np.random.normal(0, 0.1, total_length)
    
    # 1. Simulate the sync clap at second 5
    # Let's pretend Mic 2 started recording 200 samples late, Mic 3 started 500 samples early
    clap_idx = fs * 5
    mic_1[clap_idx : clap_idx+50] += 5.0
    mic_2[(clap_idx - 200) : (clap_idx - 200) + 50] += 5.0 
    mic_3[(clap_idx + 500) : (clap_idx + 500) + 50] += 5.0 
    
    # 2. Simulate the target noise at second 25
    # The physical distance adds a true delay of +300 samples to Mic 2, and +800 to Mic 3
    target_idx = fs * 25
    mic_1[target_idx : target_idx+100] += 3.0
    
    # The target in Mic 2 has the true physical delay (+300) AND the recording offset (-200)
    mic_2[(target_idx - 200 + 300) : (target_idx - 200 + 300) + 100] += 3.0 
    
    # The target in Mic 3 has the true physical delay (+800) AND the recording offset (+500)
    mic_3[(target_idx + 500 + 800) : (target_idx + 500 + 800) + 100] += 3.0 
    
    # Initialize and run
    localizer = GPUAcousticLocalizer(fs=44100)
    
    results = localizer.process_timeline(
        mic_1, mic_2, mic_3, 
        calib_window=(0, 10),   # Look for the clap here
        target_window=(20, 30)  # Look for the target here
    )
    
    print("\n--- FINAL RESULTS ---")
    for pair, data in results.items():
        print(f"{pair}: {data['samples']} samples ({data['seconds']:.6f} s)")