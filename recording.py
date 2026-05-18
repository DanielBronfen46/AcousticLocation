import time

import sounddevice as sd
import numpy as np
import soundfile as sf


# --- Configuration ---
FS = 48000  # Sample rate (44100 Hz is standard for audio)
DURATION = 15  # Duration of recording in seconds
OUTPUT_FOLDER = "mic_output"
USE_VOICEMEETER = True

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

print(sd.query_devices())
MIC1_NAME = "Microphone (USB PnP Sound Device)"
MIC2_NAME = "Microphone (USBAudio2.0)"
MIC1_ID = get_device_id_by_name(MIC1_NAME, hostapi_name="Windows WASAPI")
MIC2_ID = get_device_id_by_name(MIC2_NAME, hostapi_name="Windows WASAPI")


def record_two_mics_stereo_virtual_device(fs=FS, duration=DURATION):
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
    save_mic_output(audio1, "mic1_voicemeeter")
    save_mic_output(audio2, "mic2_voicemeeter")

    return audio1, audio2

def record_two_mics_directly(fs=FS, duration=DURATION, mic1_id=MIC1_ID, mic2_id=MIC2_ID):
    print(f"Recording for {duration} seconds...")

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
    save_mic_output(audio1, "mic1_output")
    save_mic_output(audio2, "mic2_output")

    return audio1, audio2

def save_mic_output(audio, title, fs=FS):
    seconds = int(time.time())
    file_name = f'{OUTPUT_FOLDER}/{title}_{seconds}.wav'
    sf.write(file_name, audio, fs)
    print(f"Saved {file_name}")



def get_two_signals(fs=FS, duration=DURATION):
    if USE_VOICEMEETER:
        audio1, audio2 = record_two_mics_stereo_virtual_device(fs=fs, duration=duration)
    else:
        audio1, audio2 = record_two_mics_directly(fs=fs, duration=duration)

    sig1 = audio1.flatten()
    sig2 = audio2.flatten()

    sig1 = sig1 / np.max(np.abs(sig1))
    sig2 = sig2 / np.max(np.abs(sig2))
    
    return sig1, sig2