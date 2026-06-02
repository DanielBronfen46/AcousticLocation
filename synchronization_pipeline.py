import numpy as np
import sounddevice as sd
import time
import soundfile as sf
import os
import threading
from scipy.interpolate import interp1d
from scipy import signal
from scipy.optimize import minimize_scalar

# Import all your perfected math and plotting functions
from recording import record_two_signals, FS, OUTPUT_FOLDER, trim_zeroes, apply_highpass_filter, load_two_wav_signals
from alignment import calculate_cross_correlation, align_signals, align_and_plot
from sine_wave_calibration import (  # Replace with the actual name of your file
    find_sample_frequency_diff_IQ,
    fix_frequency_and_phase,
    match_amplitude,
    plot_both_signals,
    plot_xy_signals, calibrate_sine_waves_and_plot
)


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
# ==========================================
def record_guided_session(clap_perf, sine_perf, target_start, total_duration, f_played, fs=FS):
    """
    Calculates live terminal countdowns based on the parameter windows.
    Runs the UI in a background thread to prevent Windows WDM-KS driver crashes,
    keeping the hardware audio streams safely on the Main Thread.
    """
    clap_start, clap_end = clap_perf
    sine_start, sine_end = sine_perf

    def print_instructions():
        print("\n🎙️  RECORDING STARTED! Follow the instructions:\n")

        if clap_start > 0:
            time.sleep(clap_start)

        print(f"--> [{clap_start}s - {clap_end}s] ACTION: Create delta functions!")
        time.sleep(clap_end - clap_start)

        print(f"--> [{clap_end}s - {sine_start}s] PREPARE: Get ready to play the {f_played}Hz Sine Wave...")
        time.sleep(sine_start - clap_end)

        print(f"--> [{sine_start}s - {sine_end}s] ACTION: Play {f_played}Hz sine wave!")
        time.sleep(sine_end - sine_start)

        print(f"--> [{sine_end}s - {target_start}s] PREPARE: Stop the sine wave. Get ready to speak.")
        time.sleep(target_start - sine_end)

        print(f"--> [{target_start}s - {total_duration}s] ACTION: Recording the actual target audio...")
        # The main thread blocks while recording, so we don't strictly need a final sleep here.

    # 1. Start the text UI in a background thread
    # Setting daemon=True ensures that if the hardware crashes, this thread dies instantly
    # instead of forcing you to wait 40 seconds for the countdown to finish.
    ui_thread = threading.Thread(target=print_instructions)
    ui_thread.daemon = True
    ui_thread.start()

    # 2. Start the hardware recording on the MAIN thread
    sig1, sig2 = record_two_signals(fs=fs, duration=total_duration)

    # 3. Ensure the UI thread has completely finished before moving on
    ui_thread.join()
    print("\n✅ Recording complete! Processing pipeline starting...\n")

    return sig1, sig2


def optimize_subsample_delay(sig1, sig2, fs=FS):
    """
    Finds the exact fractional sub-sample delay between two frequency-matched signals.
    Searches within a tiny +/- 2 sample window since they are already macro-aligned.
    """
    print("Starting 1D optimization for sub-sample delay...")
    search_window_sec = 2.0 / fs  # +/- 2 samples

    n_indices = np.arange(len(sig1))
    original_m_indices = np.arange(len(sig2))

    def alignment_cost(delta_t_test):
        # Shift sig2 by a tiny fractional amount of time
        m_indices = n_indices - (delta_t_test * fs)

        # Fast linear interpolation for the optimizer loop
        interpolator = interp1d(original_m_indices, sig2, kind='linear', bounds_error=False, fill_value=0.0)
        sig2_test = interpolator(m_indices)

        # Maximize the dot product (minimize the negative)
        return -np.sum(sig1 * sig2_test)

    result = minimize_scalar(
        alignment_cost,
        bounds=(-search_window_sec, search_window_sec),
        method='bounded',
        options={'xatol': 1e-7}
    )

    if result.success:
        perfect_delta_t_s = result.x
        print(f"Micro-delay found: {perfect_delta_t_s * 1000:.4f} ms")
        return perfect_delta_t_s
    else:
        print("Optimization failed. Returning 0 delay.")
        return 0.0

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

    # Apply the global frequency fix to the entire Mic 2 array
    print(f"\nResampling global Mic 2 array to correct {delta_fs_initial:.5f} Hz drift...")
    fs2_effective = fs + delta_fs_initial
    t_true_sig2 = np.arange(len(raw_sig2)) / fs2_effective
    t_target = np.arange(len(raw_sig1)) / fs

    interpolator = interp1d(t_true_sig2, raw_sig2, kind='cubic', bounds_error=False, fill_value=0.0)
    sig2_freq_fixed = interpolator(t_target)

    print("\n--- STAGE 2: MACRO TIME ALIGNMENT (CROSS-CORRELATION) ---")
    if use_global_cross_correlation:
        print("Using global cross-correlation on the entire signals...")
        corr_target1 = raw_sig1
        corr_target2 = sig2_freq_fixed
    else:
        print("Using targeted cross-correlation on the clap windows...")
        corr_target1 = slice_audio(raw_sig1, clap_analysis_window[0], clap_analysis_window[1], fs)
        corr_target2 = slice_audio(sig2_freq_fixed, clap_analysis_window[0], clap_analysis_window[1], fs)

    # Use your wrapper to align the targets and plot the results
    _, _, lag_in_samples = align_and_plot(corr_target1, corr_target2)

    # Apply the integer macro shift to the full frequency-fixed arrays
    sig1_macro, sig2_macro = align_signals(raw_sig1, sig2_freq_fixed, lag_in_samples)

    print("\n--- STAGE 3: MICRO TIME ALIGNMENT (PERFECTING THE OFFSET) ---")
    # Slice the newly aligned sine waves
    sine1_macro = slice_audio(sig1_macro, sine_analysis_window[0], sine_analysis_window[1], fs)
    sine2_macro = slice_audio(sig2_macro, sine_analysis_window[0], sine_analysis_window[1], fs)

    # 1. Clean the rumble so the optimizer only looks at pure tones
    sine1_clean = apply_highpass_filter(sine1_macro, 15.0, fs=fs)
    sine2_clean = apply_highpass_filter(sine2_macro, 15.0, fs=fs)

    # 2. Find the perfect sub-sample fractional delay
    delta_t_s = optimize_subsample_delay(sine1_clean, sine2_clean, fs=fs)

    # 3. Calculate the true amplitude scale factor on the clean sines
    # We apply the delta_t_s (with 0.0 frequency drift) so the RMS is perfectly phase-aligned
    _, sine2_micro_aligned = fix_frequency_and_phase(sine1_clean, sine2_clean, delta_fs=0.0, delta_t_s=delta_t_s, fs=fs)
    _, scale_factor = match_amplitude(sine1_clean, sine2_micro_aligned, method='rms')

    print("\n--- STAGE 4: TARGET AUDIO PROCESSING ---")
    # Extract the final target audio from the macro-aligned arrays
    target1 = slice_audio(sig1_macro, target_start_time, None, fs)
    target2 = slice_audio(sig2_macro, target_start_time, None, fs)

    # Apply the final sub-sample micro-delay (Pass 0.0 for delta_fs because we fixed it globally in Stage 1!)
    target1_fixed, target2_fixed = fix_frequency_and_phase(target1, target2, delta_fs=0.0, delta_t_s=delta_t_s, fs=fs)

    # Scale amplitude
    target2_scaled = target2_fixed * scale_factor

    # Final Validation and Save
    plot_both_signals(target1_fixed, target2_scaled, title="Final Synced Target Audio")
    plot_xy_signals(target1_fixed, target2_scaled, title="XY Plot: Target Audio Phase Verification")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    sf.write(f"{OUTPUT_FOLDER}/FINAL_mic1.wav", target1_fixed, fs)
    sf.write(f"{OUTPUT_FOLDER}/FINAL_mic2.wav", target2_scaled, fs)
    print("\n🎉 Master Flow Complete!")

    return target1_fixed, target2_scaled

def record_and_run_synchronization_pipeline(
        total_duration=40,
        f_played=200,
        clap_perf_window=(0.0, 5.0),
        clap_analysis_window=(0.0, 5.0),
        sine_perf_window=(10.0, 25.0),
        sine_analysis_window=(10.5, 24.5),
        target_start_time=30,
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

def load_and_run_synchronization_pipeline(filedesc,
                                          f_played=200,
                                          clap_analysis_window=(0.0, 5.0),
                                          sine_analysis_window=(10.5, 24.5),
                                          target_start_time=30,
                                          ):

    # Step A: Record the raw master file
    raw_sig1, raw_sig2 = load_two_wav_signals(filedesc)

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

    filedesc = filedesc2
    #load_and_run_synchronization_pipeline(filedesc)
    record_and_run_synchronization_pipeline(
                total_duration=40,
        f_played=200,
        sine_perf_window=(0.0, 15.0),
        sine_analysis_window=(0.0, 15.0),
        clap_perf_window=(18.0, 25.0),
        clap_analysis_window=(18.5, 24.5),
        target_start_time=30,
        fs=FS
    )


if __name__ == "__main__":
    main()