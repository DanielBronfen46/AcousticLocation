import os
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
# pyrefly: ignore [missing-import]
import soundfile as sf

FS = 44100
OUTPUT_FOLDER = "audio_recordings"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def save_audio_file(audio, prefix, suffix="", fs=FS):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    date_folder = timestamp.split('_')[0]
    out_dir = f"{OUTPUT_FOLDER}/{date_folder}/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    file_name = f'{out_dir}/{prefix}_{timestamp}_{suffix}.wav'
    sf.write(file_name, audio, fs)
    print(f"Saved {file_name}")


def find_ffmpeg_executable():
    """Return the path to an FFmpeg executable, searching PATH and common Windows install locations."""
    ffmpeg_exe = shutil.which('ffmpeg')
    if ffmpeg_exe:
        return ffmpeg_exe

    env_path = os.environ.get('FFMPEG_PATH')
    if env_path and Path(env_path).exists():
        return env_path

    local_appdata = Path(os.environ.get('LOCALAPPDATA', ''))
    search_bases = []
    if local_appdata.exists():
        search_bases.extend([
            local_appdata / 'Microsoft' / 'WindowsApps',
            local_appdata / 'Microsoft' / 'WinGet' / 'Packages'
        ])

    for base in search_bases:
        if base.exists():
            for candidate in base.rglob('ffmpeg.exe'):
                return str(candidate)

    for prog_root in [Path('C:/Program Files/ffmpeg'), Path('C:/Program Files (x86)/ffmpeg')]:
        candidate = prog_root / 'bin' / 'ffmpeg.exe'
        if candidate.exists():
            return str(candidate)

    return None

def convert_m4a_to_wav(input_m4a_path, output_wav_path=None, duration_sec=None, fs=None):
    """Convert an M4A file to WAV while preserving original sample rate and duration by default."""
    input_path = Path(input_m4a_path)
    if not input_path.is_absolute():
        input_path = Path(OUTPUT_FOLDER) / input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_wav_path is None:
        output_path = input_path.with_suffix('.wav')
    else:
        output_path = Path(output_wav_path)
        if not output_path.is_absolute():
            output_path = Path(OUTPUT_FOLDER) / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = find_ffmpeg_executable()
    if ffmpeg_exe is None:
        raise RuntimeError(
            "FFmpeg is required to convert m4a to wav. Install FFmpeg and ensure it is on your PATH, or set FFMPEG_PATH."
        )

    command = [ffmpeg_exe, '-y', '-i', str(input_path)]
    if duration_sec is not None:
        command += ['-t', str(duration_sec)]
    if fs is not None:
        command += ['-ar', str(fs)]
    command += [str(output_path)]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg conversion failed:\n{result.stderr.strip()}"
        )

    info = []
    if duration_sec is not None:
        info.append(f"{duration_sec}s")
    if fs is not None:
        info.append(f"{fs} Hz")
    info_str = f" ({', '.join(info)})" if info else ""

    print(f"Converted '{input_path}' -> '{output_path}'{info_str}")
    return str(output_path)


def match_signal_length(sig1, sig2):
    min_len = min(len(sig1), len(sig2))
    sig1_matched = sig1[:min_len]
    sig2_matched = sig2[:min_len]

    return sig1_matched, sig2_matched

def match_signals_and_trim_zeroes(sig1, sig2, fs=FS):
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

    print(f"Trimmed {start_idx/fs} seconds from the start.")
    print(f"Trimmed {(len(s1) - (end_idx + 1))/fs} seconds from the end.")

    return sig1_cropped, sig2_cropped

def match_n_signals_and_trim_zeroes(signals, fs=FS):
    """
    Finds the exact start and end of the valid data across N signals
    (ignoring 0.0 padding) and crops ALL signals to the shared overlapping window.
    Safely handles length mismatches and zero-padding on any signal.
    """
    if not signals:
        return []

    # 1. Guarantee a 1:1 time mapping by truncating all to the shortest length
    # This replaces the 2-signal match_signal_length function
    min_len = min(len(sig) for sig in signals)
    matched_signals = [sig[:min_len] for sig in signals]

    starts = []
    ends = []

    # 2 & 3. Find the valid indices and independent bounds for each signal
    for i, sig in enumerate(matched_signals):
        valid_indices = np.where(sig != 0.0)[0]

        # Failsafe: if any array is completely empty/zeros
        if len(valid_indices) == 0:
            print(f"Warning: Signal {i} is entirely zeros. Aborting trim.")
            return matched_signals

        starts.append(valid_indices[0])
        ends.append(valid_indices[-1])

    # 4. Calculate the shared overlapping window
    # Start when ALL have started, stop when ANY stops
    start_idx = max(starts)
    end_idx = min(ends)

    # Failsafe: if they somehow don't overlap at all
    if start_idx > end_idx:
        print("Warning: Signals have no overlapping valid data.")
        return matched_signals

    # 5. Crop all signals using the shared valid indices
    cropped_signals = [sig[start_idx : end_idx + 1] for sig in matched_signals]

    print(f"Trimmed {start_idx/fs:.5f} seconds from the start.")
    print(f"Trimmed {(min_len - (end_idx + 1))/fs:.5f} seconds from the end.")

    return cropped_signals

def load_wav_signal(file_desc, file_name):
    """
    Reads a WAV file and returns the signal and its sampling frequency.
    Ensures the output is a 1D numpy array (mono) for mathematical operations.
    """
    date_folder = file_desc.split('_')[0]
    nested_path = f"{OUTPUT_FOLDER}/{date_folder}/{file_desc}/{file_name}"
    flat_path = f"{OUTPUT_FOLDER}/{file_name}"
    
    if os.path.exists(nested_path):
        target_path = nested_path
    elif os.path.exists(flat_path):
        target_path = flat_path
    else:
        target_path = nested_path
        
    # Read the audio data and sampling frequency
    signal, fs = sf.read(target_path)

    # Check if the signal is multi-channel (e.g., stereo)
    if len(signal.shape) > 1:
        # Average the channels together to create a single 1D array
        signal = np.mean(signal, axis=1)

    return signal, fs

def load_two_wav_files(file_desc):
    mic1_filename = f"{file_desc}_mic1.wav"
    mic2_filename = f"{file_desc}_mic2.wav"

    sig1, fs1 = load_wav_signal(file_desc, mic1_filename)
    sig2, fs2 = load_wav_signal(file_desc, mic2_filename)

    freq_lst = [fs1, fs2]
    frequency_different_from_FS = next((i for i, item in enumerate(freq_lst) if item != FS), -1)
    if frequency_different_from_FS != -1:
        raise ValueError(f"Mic {frequency_different_from_FS+1} has sampling rate {freq_lst[frequency_different_from_FS]}, instead of FS={FS}.")

    sig1, sig2 = match_signals_and_trim_zeroes(sig1, sig2)

    return sig1, sig2

def load_n_wav_files(file_desc, n=None):
    """
    Loads N wav files following the naming convention '{file_desc}_micX.wav'.
    If 'n' is not provided, it will automatically detect and load all matching files.
    """
    signals = []
    i = 1

    while True:
        # If n is specified, stop when we exceed n
        if n is not None and i > n:
            break

        mic_filename = f"{file_desc}_mic{i}.wav"

        # If n is NOT specified, stop when we can't find the next consecutive file
        if n is None:
            date_folder = file_desc.split('_')[0]
            nested_path = f"{OUTPUT_FOLDER}/{date_folder}/{file_desc}/{mic_filename}"
            flat_path = f"{OUTPUT_FOLDER}/{mic_filename}"
            if not (os.path.exists(nested_path) or os.path.exists(flat_path)):
                if i == 1:
                    raise FileNotFoundError(f"Error: Could not find the starting file '{mic_filename}'")
                break

        # Load the signal (assuming load_wav_signal is imported/available)
        sig, fs = load_wav_signal(file_desc, mic_filename)

        # Ensure all sampling frequencies perfectly match the first file

        if fs != FS:
            raise ValueError(f"Sampling frequency mismatch at {mic_filename} ({fs} Hz vs base {FS} Hz).")


        signals.append(sig)
        i += 1


    signals = match_n_signals_and_trim_zeroes(signals)

    print(f"Loaded {len(signals)} microphone signals for '{file_desc}'")

    return signals