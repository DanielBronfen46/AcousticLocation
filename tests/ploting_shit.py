import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import os
import glob
import argparse


def plot_wav_spectrogram(wav_path, nfft=1024, noverlap=None, cmap='viridis'):
    """Plot a spectrogram from a WAV file.

    Args:
        wav_path (str): Path to the WAV file.
        nfft (int, optional): Window length for FFT. Defaults to 1024.
        noverlap (int, optional): Number of overlapping samples. Defaults to nfft // 2.
        cmap (str, optional): Matplotlib colormap. Defaults to 'viridis'.
    """
    sample_rate, samples = wavfile.read(wav_path)

    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)

    if noverlap is None:
        noverlap = nfft // 2

    frequencies, times, Sxx = spectrogram(
        samples,
        fs=sample_rate,
        window='hann',
        nperseg=nfft,
        noverlap=noverlap,
        scaling='density',
        mode='magnitude',
    )

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap=cmap)
    plt.colorbar(label='Intensity [dB]')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title(f'Spectrogram of {wav_path}')
    plt.tight_layout()
    plt.show()


def find_latest_wav(directory='.'):
    """Return path to the most recently modified .wav file in directory."""
    files = glob.glob(os.path.join(directory, '*.wav'))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot spectrogram for a WAV file. If no file is given, uses the latest .wav in the directory.')
    parser.add_argument('wav', nargs='?', help='Path to a WAV file')
    parser.add_argument('--dir', '-d', default='audio_recordings', help='Directory to search for latest WAV when no file provided')
    args = parser.parse_args()

    wav_path = args.wav or find_latest_wav(args.dir)
    if wav_path is None:
        print('No .wav files found.')
    else:
        print(f'Using WAV: {wav_path}')
        plot_wav_spectrogram(wav_path)

