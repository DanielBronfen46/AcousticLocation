import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
# pyrefly: ignore [missing-import]
import torch



def plot_wav_spectrogram(wav_path, accuracy=50, cmap='viridis'):
    """Plot a spectrogram from a WAV file using PyTorch (GPU if available).

    Args:
        wav_path (str): Path to the WAV file.
        accuracy (int, optional): Time-frequency accuracy from 0 to 100. Defaults to 50.
        cmap (str, optional): Matplotlib colormap. Defaults to 'viridis'.
    """
    sample_rate, samples = wavfile.read(wav_path)

    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)

    # Clamp accuracy between 0 and 100
    accuracy = max(0, min(100, accuracy))
    
    # Scale nfft based on accuracy (0 -> 256, 100 -> 4096)
    nfft_vals = [256, 512, 1024, 2048, 4096]
    idx = int((accuracy / 100.0) * 4.99)
    nfft = nfft_vals[idx]
    
    # Scale hop length (higher accuracy = more overlap = finer time resolution)
    # Factor ranges from 2 to 8 (so hop_length goes from nfft//2 down to nfft//8)
    overlap_factor = 2 + int((accuracy / 100.0) * 6)
    hop_length = nfft // overlap_factor

    # Automatically select the best available hardware accelerator
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Calculating spectrogram on {device} with accuracy={accuracy} (nfft={nfft}, hop_length={hop_length})")

    tensor_samples = torch.tensor(samples, dtype=torch.float32, device=device)
    window = torch.hann_window(nfft, device=device)
    
    stft = torch.stft(
        tensor_samples, 
        n_fft=nfft, 
        hop_length=hop_length, 
        win_length=nfft, 
        window=window, 
        return_complex=True
    )
    
    Sxx = torch.abs(stft).cpu().numpy()
    
    frequencies = np.linspace(0, sample_rate / 2, Sxx.shape[0])
    times = np.linspace(0, len(samples) / sample_rate, Sxx.shape[1])

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap=cmap)
    plt.colorbar(label='Intensity [dB]')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title(f'Spectrogram of {wav_path} (accuracy: {accuracy})')
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    wav_path = "audio_recordings/2026-06-18/2026-06-18_23-25-42/2026-06-18_23-25-42_mic1.wav"
    plot_wav_spectrogram(wav_path, accuracy=100)

