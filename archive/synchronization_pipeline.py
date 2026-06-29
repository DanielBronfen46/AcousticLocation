import numpy as np
import time
import os
import threading
from archive.recording import save_audio_file

# Import all your perfected math and plotting functions
from archive.recording import FS, OUTPUT_FOLDER
from alignment import align_signals_given_lag, align_two_signals
from plotting_functions import plot_both_signals
from sine_wave_calibration import (  # Replace with the actual name of your file
    match_amplitude,
    plot_xy_signals, calibrate_sine_waves_and_plot
)
from sound_file_handling import convert_m4a_to_wav, load_two_wav_files


# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def sec_to_idx(seconds, fs=FS):
    """Converts a time in seconds to an array index."""
    return int(seconds * fs)

def slice_audio(sig, start_sec, end_sec=None, fs=FS):
    """Slices an audio array safely based on seconds."""
    start_idx = sec_to_idx(start_sec, fs)
    if end_sec is None:
        return sig[start_idx:]
    end_idx = sec_to_idx(end_sec, fs)
    return sig[start_idx:end_idx]


# ==========================================
# 3. GUIDED RECORDING FUNCTION
# =====================================
def record_guided_session(clap_perf, sine_perf, target_start, total_duration, f_played, fs=FS):
    """
    Calculates live terminal countdowns based on the parameter windows.
    Runs the UI in a background thread to prevent Windows WDM-KS driver crashes,
    keeping the hardware audio streams safely on the Main Thread.
    """

    # 1. Start the text UI in a background thread
    # Setting daemon=True ensures that if the hardware crashes, this thread dies instantly
    # instead of forcing you to wait 40 seconds for the countdown to finish.
    ui_thread = threading.Thread(target=print_instructions,
                                 args=(clap_perf, sine_perf, target_start, total_duration, f_played))
    ui_thread.daemon = True
    ui_thread.start()

    # 2. Start the hardware recording on the MAIN thread
    #sig1, sig2 = record_two_signals(fs=fs, duration=total_duration)
    sig1, sig2 = None, None

    # 3. Ensure the UI thread has completely finished before moving on
    ui_thread.join()
    print("\n✅ Recording complete! Processing pipeline starting...\n")

    return sig1, sig2

def print_instructions(clap_perf, sine_perf, target_start, total_duration, f_played):
    clap_start, clap_end = clap_perf
    sine_start, sine_end = sine_perf


    print("\n🎙️  RECORDING STARTED! Follow the instructions:\n")

    # 1. Wait for Sine Wave start
    if sine_start > 0:
        time.sleep(sine_start)

    # 2. Sine Wave Action
    print(f"--> [{sine_start}s - {sine_end}s] ACTION: Play {f_played}Hz sine wave!")
    time.sleep(sine_end - sine_start)

    # 3. Prepare for Claps
    print(f"--> [{sine_end}s - {clap_start}s] PREPARE: Stop the sine wave. Get ready to clap...")
    time.sleep(clap_start - sine_end)

    # 4. Claps Action (Delta functions)
    print(f"--> [{clap_start}s - {clap_end}s] ACTION: Create delta functions (Clap)!")
    time.sleep(clap_end - clap_start)

    # 5. Prepare for Target Audio
    print(f"--> [{clap_end}s - {target_start}s] PREPARE: Stop clapping. Get ready to speak.")
    time.sleep(target_start - clap_end)

    # 6. Target Audio Action
    print(f"--> [{target_start}s - {total_duration}s] ACTION: Recording the actual target audio...")
    time.sleep(total_duration - target_start)

    print("--> STOP")

def run_synchronization_pipeline(
        raw_sig1, raw_sig2,
        f_played,
        clap_analysis_window,
        sine_analysis_window,
        target_start_time,
        fs=FS,
        use_global_cross_correlation=False
):
    from scipy.interpolate import interp1d

    print("\n--- STAGE 1: INITIAL FREQUENCY FIT ---")
    # Slice the raw, unaligned arrays.
    sine1_raw = slice_audio(raw_sig1, sine_analysis_window[0], sine_analysis_window[1], fs)
    sine2_raw = slice_audio(raw_sig2, sine_analysis_window[0], sine_analysis_window[1], fs)

    # Fit the sines to find the initial frequency offset.
    _, _, delta_fs_initial, _, _ = calibrate_sine_waves_and_plot(sine1_raw, sine2_raw, f_played, fs)

    delta_fs_initial = delta_fs_initial
    # Apply the global frequency fix to the entire Mic 2 array
    print(f"\nResampling global Mic 2 array to correct {delta_fs_initial:.5f} Hz drift...")
    fs2_effective = fs + delta_fs_initial
    t_true_sig2 = np.arange(len(raw_sig2)) / fs2_effective
    t_target = np.arange(len(raw_sig1)) / fs

    interpolator = interp1d(t_true_sig2, raw_sig2, kind='cubic', bounds_error=False, fill_value=0.0)
    sig2_freq_fixed = interpolator(t_target)

    print("\n--- STAGE 2: MACRO TIME ALIGNMENT (CROSS-CORRELATION AND AMPLITUDE) ---")
    if use_global_cross_correlation:
        print("Using global cross-correlation on the entire signals...")
        corr_target1 = raw_sig1
        corr_target2 = sig2_freq_fixed
    else:
        print("Using targeted cross-correlation on the clap windows...")
        corr_target1 = slice_audio(raw_sig1, clap_analysis_window[0], clap_analysis_window[1], fs)
        corr_target2 = slice_audio(sig2_freq_fixed, clap_analysis_window[0], clap_analysis_window[1], fs)

    # Use your wrapper to align the targets and plot the results
    _, _, lag_in_samples = align_two_signals(corr_target1, corr_target2)

    # Apply the integer macro shift to the full frequency-fixed arrays
    sig1_macro, sig2_macro = align_signals_given_lag(raw_sig1, sig2_freq_fixed, lag_in_samples)

    # 3. Calculate the true amplitude scale factor on the clean sines
    _, scale_factor = match_amplitude(sine1_raw, sine2_raw, method='rms')

    print("\n--- STAGE 3: TARGET AUDIO PROCESSING ---")
    # Extract the final target audio from the macro-aligned arrays
    target1_fixed = slice_audio(sig1_macro, target_start_time, None, fs)
    target2 = slice_audio(sig2_macro, target_start_time, None, fs)

    # Scale amplitude
    target2_fixed = target2 * scale_factor

    # Final Validation and Save
    plot_both_signals(target1_fixed, target2_fixed, title="Final Synced Target Audio")
    plot_xy_signals(target1_fixed, target2_fixed, title="XY Plot: Target Audio Phase Verification")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    save_audio_file(target1_fixed, prefix="FINAL", suffix="mic1", fs=fs)
    save_audio_file(target2_fixed, prefix="FINAL", suffix="mic2", fs=fs)
    print("\n🎉 Master Flow Complete!")

    return target1_fixed, target2_fixed

def record_and_run_synchronization_pipeline(
        total_duration,
        f_played,
        sine_perf_window,
        sine_analysis_window,
        clap_perf_window,
        clap_analysis_window,
        target_start_time,
        fs=FS
):
    # Step A: Record the raw master file
    raw_sig1, raw_sig2 = record_guided_session(
        clap_perf=clap_perf_window,
        sine_perf=sine_perf_window,
        target_start=target_start_time,
        total_duration=total_duration,
        f_played=f_played,
        fs=fs
    )

    run_synchronization_pipeline(
        raw_sig1, raw_sig2,
        f_played=f_played,
        clap_analysis_window=clap_analysis_window,
        sine_analysis_window=sine_analysis_window,
        target_start_time=target_start_time,
        fs=FS
    )

def load_and_run_synchronization_pipeline(
        filedesc,    #             total_duration=40,
        f_played=200,
        sine_analysis_window=(0.0, 15.0),
        clap_analysis_window=(18.5, 24.5),
        target_start_time=30
                                          ):

    # Step A: Record the raw master file
    raw_sig1, raw_sig2 = load_two_wav_files(filedesc)

    run_synchronization_pipeline(
        raw_sig1, raw_sig2,
        f_played=f_played,
        clap_analysis_window=clap_analysis_window,
        sine_analysis_window=sine_analysis_window,
        target_start_time=target_start_time,
        fs=FS
    )


def main():
    filedesc1 = "recording_2026-05-31_19-19-30_40s"
    filedesc2 = "recording_2026-05-31_21-20-30_40s"
    filedesc3 = "recording_2026-06-02_10-24-45_40s"
    filedesc4 = "recording_outside1_40s"
    
    filedesc = "telephone3"

    total_duration = 40
    f_played = 200
    sine_perf_window = (0.0, 18.0)
    sine_analysis_window = (3.0, 15.0)
    clap_perf_window = (20.0, 35.0)
    clap_analysis_window = (21.0, 34.0)
    target_start_time = 35


    load_and_run_synchronization_pipeline(filedesc,
                                          sine_analysis_window=sine_analysis_window,
        clap_analysis_window=clap_analysis_window,
       target_start_time=target_start_time)

    # record_and_run_synchronization_pipeline(
    #             total_duration=total_duration,
    #     f_played=f_played,
    #     sine_perf_window=sine_perf_window,
    #     sine_analysis_window=sine_analysis_window,
    #     clap_perf_window=clap_perf_window,
    #     clap_analysis_window=clap_analysis_window,
    #     target_start_time=target_start_time,
    #     fs=FS
    # )

    #print_instructions(clap_perf_window, sine_perf_window, target_start_time, total_duration, f_played)


if __name__ == "__main__":
    convert_m4a_to_wav("telephone3_mic1.m4a", "telephone3_mic1.wav",fs=48000)
    convert_m4a_to_wav("telephone3_mic2.m4a", "telephone3_mic2.wav",fs=48000)
    main()

