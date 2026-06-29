import os

import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt

from sound_file_handling import save_audio_file

# --- Configuration ---
FS = 44100  # Sample rate (44100 Hz is standard for audio)
OUTPUT_FOLDER = "audio_recordings"

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
    save_audio_file(audio1, "asio", "mic1")
    save_audio_file(audio2, "asio", "mic2")

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
    save_audio_file(audio1, "voicemeeter", "mic1")
    save_audio_file(audio2, "voicemeeter", "mic2")

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
    save_audio_file(audio1, "recording", f"{duration}s_mic1")
    save_audio_file(audio2, "recording", f"{duration}s_mic2")

    return audio1, audio2
