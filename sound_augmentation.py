import numpy as np
import scipy.io.wavfile as wavfile
import random
import os

def add_noise_and_echo(input_path: str, output_path: str, 
                       snr_db: float = 20.0, 
                       echo_delay_ms: tuple = (5.0, 30.0), 
                       echo_attenuation: tuple = (0.2, 0.5)):
    """
    Reads a WAV file, adds a random small room echo, adds Gaussian white noise,
    and saves the augmented audio to a new file.

    Args:
        input_path: Path to the original .wav file.
        output_path: Path to save the augmented .wav file.
        snr_db: Desired Signal-to-Noise Ratio in dB for the added noise.
        echo_delay_ms: Tuple of (min, max) delay for the echo in milliseconds.
                       (5-30ms represents reflections in an average room).
        echo_attenuation: Tuple of (min, max) amplitude multiplier for the echo.
    """
    # 1. Load the audio file
    fs, audio = wavfile.read(input_path)
    
    # Handle stereo/multi-channel by converting to float32
    # If the audio is integer, normalize to [-1.0, 1.0]
    is_int16 = audio.dtype == np.int16
    if is_int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)

    # 2. Add Echo
    # Pick a random delay and attenuation
    delay_ms = random.uniform(echo_delay_ms[0], echo_delay_ms[1])
    attenuation = random.uniform(echo_attenuation[0], echo_attenuation[1])
    
    delay_samples = int((delay_ms / 1000.0) * fs)
    
    # Create an empty array for the echo, padded with zeros at the start
    if audio.ndim == 1:
        echo = np.zeros(len(audio) + delay_samples, dtype=np.float32)
        echo[delay_samples:] = audio * attenuation
        
        # Pad the original audio to match the new length
        padded_audio = np.zeros_like(echo)
        padded_audio[:len(audio)] = audio
    else:
        # Multi-channel support
        echo = np.zeros((len(audio) + delay_samples, audio.shape[1]), dtype=np.float32)
        echo[delay_samples:, :] = audio * attenuation
        
        padded_audio = np.zeros_like(echo)
        padded_audio[:len(audio), :] = audio
        
    augmented_audio = padded_audio + echo

    # 3. Add White Noise
    # Calculate signal power based on the PEAK (since the audio is 99% silence)
    # Using the RMS power of a sine wave with the same peak: (peak / sqrt(2))^2
    peak_amp = np.max(np.abs(augmented_audio))
    signal_power = (peak_amp / 1.414) ** 2
    
    # SNR(dB) = 10 * log10(signal_power / noise_power)
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Generate Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), augmented_audio.shape).astype(np.float32)
    augmented_audio += noise

    # 4. Normalize and Save
    # Prevent clipping by normalizing if the peak exceeds 1.0
    peak = np.max(np.abs(augmented_audio))
    if peak > 1.0:
        augmented_audio = augmented_audio / peak

    # Convert back to int16 if the original was int16
    if is_int16:
        augmented_audio = (augmented_audio * 32767.0).astype(np.int16)

    wavfile.write(output_path, fs, augmented_audio)
    
    print(f"Augmented file saved to: {output_path}")
    print(f"  - Echo applied: delay = {delay_ms:.1f} ms, attenuation = {attenuation:.2f}")
    print(f"  - Noise applied: SNR = {snr_db} dB")


def augment_dataset(input_dir: str = "audio_recordings", output_dir: str = "audio_recordings_augmented"):
    """
    Iterates through the input directory structure and creates an identical
    structure in the output directory, augmenting every .wav file found.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    # Walk through the directory tree
    for root, dirs, files in os.walk(input_dir):
        # Calculate the relative path to maintain structure
        rel_path = os.path.relpath(root, input_dir)
        target_dir = os.path.join(output_dir, rel_path)
        
        # Ensure the target directory exists
        os.makedirs(target_dir, exist_ok=True)
        
        for file in files:
            input_file = os.path.join(root, file)
            output_file = os.path.join(target_dir, file)
            
            if file.endswith(".wav"):
                print(f"Processing: {input_file}")
                try:
                    add_noise_and_echo(input_file, output_file, snr_db=20.0)
                except Exception as e:
                    print(f"  -> Failed to augment {file}: {e}")
            else:
                # For non-wav files (like .txt metadata), just copy them over
                import shutil
                shutil.copy2(input_file, output_file)

if __name__ == "__main__":
    print("Starting dataset augmentation (Corrected Noise)...")
    augment_dataset()
    print("Dataset augmentation complete!")
