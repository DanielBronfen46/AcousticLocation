import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
SAMPLE_RATE = 44100  # Standard audio sampling frequency (Hz)
DURATION = 7.0       # Recording duration in seconds

print("🎤 Recording starting in 3... 2... 1...")
print("👉 SPEAK NOW!")

# Record audio as a floating-point NumPy array
# channels=1 ensures mono recording, which is ideal for voice analysis
audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()  # Wait until the 5-second recording is completely finished

print("🛑 Recording finished. Processing spectrogram...")

# Squeeze the array to make it 1D for matplotlib
audio_signal = np.squeeze(audio_data)

# --- Plotting ---
plt.figure(figsize=(12, 6))

# Generate the spectrogram
# NFFT: Window size for the FFT. 1024 gives a great balance between time and frequency resolution for speech.
# noverlap: Higher overlap creates a smoother visual gradient.
frequencies, times, spectrogram, im = plt.specgram(
    audio_signal, 
    NFFT=4096,         # Increased from 1024 to 4096
    Fs=SAMPLE_RATE, 
    noverlap=3072,     # Increased to maintain a smooth overlap
    window=np.blackman(4096),  # Using a Blackman window to reduce spectral leakage
    cmap='viridis'
)




# Customize the chart layout
plt.title("Wideband Spectrogram of Human Speech", fontsize=14, fontweight='bold')
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Frequency (Hz)", fontsize=12)

# Limit the vertical axis to focus on the human voice range (0 to 8000 Hz is plenty)
plt.ylim(0, 8000)  

# Add a color bar to show intensity/volume (dB scale representation)
cbar = plt.colorbar(im)
cbar.set_label("Intensity (Logarithmic Scale)", fontsize=11)

plt.tight_layout()
print("📊 Displaying plot...")
plt.show()