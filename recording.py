import os


from datetime import datetime

import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt

# --- Configuration ---
FS = 48000  # Sample rate (44100 Hz is standard for audio)
OUTPUT_FOLDER = "mic_output"
USE_VOICEMEETER = False

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def record_asio(duration, fs=FS):
    os.environ["SD_ENABLE_ASIO"] = "1"
    # Find ASIO4ALL
    try:
        asio_id = next(i for i, d in enumerate(sd.query_devices()) if "ASIO4ALL" in d['name'])
    except StopIteration:
        raise ValueError("ASIO4ALL not found.")

    print(f"Recording mics from asio for {duration} seconds...")

    # We MUST ask for 4 channels here because both mics are stereo
    quad_audio = sd.rec(int(duration * fs), samplerate=fs, channels=4, device=asio_id)
    sd.wait()

    # Slice the arrays.
    # Grab Channel 0 (Index 0) for Mic 1
    # Grab Channel 3 (Index 2) for Mic 2
    audio1 = quad_audio[:, 0]
    audio2 = quad_audio[:, 2]

    # Save them
    save_mic_output(audio1, "asio", "mic1")
    save_mic_output(audio2, "asio", "mic2")

    print("Recording complete! You now have both mics.")
    return audio1, audio2

def get_device_id_by_name(device_name, hostapi_name):
    """
    Searches the system for an audio device by its name and API,
    and returns its dynamic ID number.
    """
    devices = sd.query_devices()

    # We use sounddevice's hostapi list to make sure we match the exact driver type
    hostapis = sd.query_hostapis()
    target_api_index = -1

    for idx, api in enumerate(hostapis):
        if hostapi_name in api['name']:
            target_api_index = idx
            break

    for idx, device in enumerate(devices):
        # Check if the name matches, the API matches, and it has input channels
        if (device_name in device['name'] and
                device['hostapi'] == target_api_index and
                device['max_input_channels'] > 0):
            return idx

    # If it can't find it, raise an error so you know immediately
    raise ValueError(f"Could not find an input device matching '{device_name}' on {hostapi_name}.")

def record_two_mics_stereo_virtual_device(duration, fs=FS):
    VOICEMEETER_NAME = "Voicemeeter Out B1"
    VOICEMEETER_ID = get_device_id_by_name(VOICEMEETER_NAME, hostapi_name="Windows WASAPI")
    device_id = VOICEMEETER_ID

    print(f"Recording from VoiceMeeter for {duration} seconds...")

    # Notice we use channels=2 here to capture the stereo VoiceMeeter feed
    stereo_audio = sd.rec(int(duration * fs), samplerate=fs, channels=2, device=device_id)

    # sd.rec runs in the background, so we must tell Python to wait for it to finish
    sd.wait()


    # Slice the stereo array back into two separate mono arrays
    # stereo_audio[:, 0] grabs all rows from the first column (Left = Mic 1)
    # stereo_audio[:, 1] grabs all rows from the second column (Right = Mic 2)
    audio1 = stereo_audio[:, 0]
    audio2 = stereo_audio[:, 1]

    # Save to WAV files
    save_mic_output(audio1, "voicemeeter", "mic1")
    save_mic_output(audio2, "voicemeeter", "mic2")

    return audio1, audio2

def record_two_mics_directly(duration, fs=FS):
    #print(sd.query_devices())
    MIC1_NAME = "Microphone (USB PnP Sound Device)"
    MIC2_NAME = "Microphone (USBAudio2.0)"
    mic1_id = get_device_id_by_name(MIC1_NAME, hostapi_name="Windows WASAPI")
    mic2_id = get_device_id_by_name(MIC2_NAME, hostapi_name="Windows WASAPI")


    print(f"Recording mics directly for {duration} seconds...")

    # Query the devices to see how many channels they natively expect (prevents WASAPI crashes)
    ch1 = int(sd.query_devices(mic1_id)['max_input_channels'])
    ch2 = int(sd.query_devices(mic2_id)['max_input_channels'])

    # Lists to hold the audio data chunks as they come in
    recording1 = []
    recording2 = []

    # Callbacks to grab the audio data from the streams
    def callback1(indata, frames, time, status):
        if status:
            print(f"Mic 1 Status: {status}")
        recording1.append(indata.copy())

    def callback2(indata, frames, time, status):
        if status:
            print(f"Mic 2 Status: {status}")
        recording2.append(indata.copy())

    # Set up the two independent input streams using the dynamic channel variables
    stream1 = sd.InputStream(samplerate=fs, channels=ch1, device=mic1_id, callback=callback1)
    stream2 = sd.InputStream(samplerate=fs, channels=ch2, device=mic2_id, callback=callback2)

    # Open both streams simultaneously
    with stream1, stream2:
        # Keep the main thread alive while the callbacks record in the background
        sd.sleep(int(duration * 1000))

    # Combine the chunks of data into single arrays
    audio1 = np.concatenate(recording1, axis=0)
    audio2 = np.concatenate(recording2, axis=0)

    # --- NEW: Fix the output if it recorded in Stereo ---
    # If the microphone recorded 2 or more channels, it creates a 2D array.
    # We must extract just the first column (channel 0) so the math downstream doesn't break.
    if len(audio1.shape) > 1 and audio1.shape[1] > 1:
        audio1 = audio1[:, 0]

    if len(audio2.shape) > 1 and audio2.shape[1] > 1:
        audio2 = audio2[:, 0]

    # Save to WAV files (sf.write naturally saves 1D arrays as mono files)
    save_mic_output(audio1, "recording", f"{duration}s_mic1")
    save_mic_output(audio2, "recording", f"{duration}s_mic2")

    return audio1, audio2

def save_mic_output(audio, prefix, suffix="", fs=FS):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f'{OUTPUT_FOLDER}/{prefix}_{timestamp}_{suffix}.wav'
    sf.write(file_name, audio, fs)
    print(f"Saved {file_name}")


def match_signal_length(sig1, sig2):
    min_len = min(len(sig1), len(sig2))
    sig1_matched = sig1[:min_len]
    sig2_matched = sig2[:min_len]

    return sig1_matched, sig2_matched

def trim_zeroes(sig1, sig2):
    """
    Finds the exact start and end of the valid data in both signals
    (ignoring 0.0 padding) and crops BOTH signals to the shared overlapping window.
    Safely handles length mismatches and zero-padding on either signal.
    """
    # 1. Guarantee a 1:1 time mapping by truncating to the shortest length
    s1, s2 = match_signal_length(sig1, sig2)

    # 2. Find all indices where the signals have real data
    valid_indices_1 = np.where(s1 != 0.0)[0]
    valid_indices_2 = np.where(s2 != 0.0)[0]

    # Failsafe: if one array is completely empty
    if len(valid_indices_1) == 0 or len(valid_indices_2) == 0:
        print("Warning: One or both signals are entirely zeros.")
        return s1, s2

    # 3. Get the independent bounds for both signals
    start1, end1 = valid_indices_1[0], valid_indices_1[-1]
    start2, end2 = valid_indices_2[0], valid_indices_2[-1]

    # 4. Calculate the shared overlapping window
    # Start when BOTH have started, stop when EITHER stops
    start_idx = max(start1, start2)
    end_idx = min(end1, end2)

    # Failsafe: if they somehow don't overlap at all
    if start_idx > end_idx:
        print("Warning: Signals have no overlapping valid data.")
        return s1, s2

    # 5. Crop both signals using the shared valid indices
    sig1_cropped = s1[start_idx : end_idx + 1]
    sig2_cropped = s2[start_idx : end_idx + 1]

    print(f"Trimmed {start_idx/48000} seconds from the start.")
    print(f"Trimmed {(len(s1) - (end_idx + 1))/48000} seconds from the end.")

    return sig1_cropped, sig2_cropped

def apply_highpass_filter(sig, cutoff_freq, fs=FS, order=4):
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
    b, a = butter(order, normal_cutoff, btype='high', analog=False)

    # 4. Apply the filter forwards and backwards to preserve exact phase alignment
    filtered_sig = filtfilt(b, a, sig)

    return filtered_sig

def final_processing(sig1, sig2):
    sig1, sig2 = trim_zeroes(sig1, sig2)

    sig1 = apply_highpass_filter(sig1, cutoff_freq=15)
    sig2 = apply_highpass_filter(sig2, cutoff_freq=15)

    return sig1, sig2

def record_two_signals(duration, fs=FS):
    if USE_VOICEMEETER:
        audio1, audio2 = record_two_mics_stereo_virtual_device(fs=fs, duration=duration)
    else:
        audio1, audio2 = record_two_mics_directly(fs=fs, duration=duration)

    sig1 = audio1.flatten()
    sig2 = audio2.flatten()

    sig1_matched, sig2_matched = final_processing(sig1, sig2)

    return sig1_matched, sig2_matched

def load_wav_signal(file_name):
    """
    Reads a WAV file and returns the signal and its sampling frequency.
    Ensures the output is a 1D numpy array (mono) for mathematical operations.
    """
    # Read the audio data and sampling frequency
    file_name = f"{OUTPUT_FOLDER}/{file_name}"
    signal, fs = sf.read(file_name)

    # Check if the signal is multi-channel (e.g., stereo)
    if len(signal.shape) > 1:
        print(f"Warning: '{file_name}' is multi-channel. Converting to mono...")
        # Average the channels together to create a single 1D array
        signal = np.mean(signal, axis=1)

    return signal, fs

def load_two_wav_signals(file_desc):
    mic1_filename = f"{file_desc}_mic1.wav"
    mic2_filename = f"{file_desc}_mic2.wav"

    sig1, fs1 = load_wav_signal(mic1_filename)
    sig2, fs2 = load_wav_signal(mic2_filename)

    if fs1 != fs2:
        raise ValueError(f"{mic1_filename} and {mic2_filename} have different sampling frequencies")

    sig1_matched, sig2_matched = final_processing(sig1, sig2)

    return sig1_matched, sig2_matched