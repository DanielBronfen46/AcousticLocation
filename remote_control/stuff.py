import subprocess
import time
import threading
import os

# ==========================================
#               CONFIGURATION
# ==========================================
# Replace these with your actual hotspot gateway IPs (e.g., "192.168.43.1:5555")
# If tracking multiple phones on the same hotspot, add their specific IPs here.
DEVICES = ["10.40.206.172:5555"]  

RECORD_DURATION = 10       # Duration of the video in seconds
LOCAL_SAVE_DIR = r"C:\PhoneRecordings"

# ==========================================
#             AUTOMATION LOGIC
# ==========================================

def run_adb(device_target, args):
    """Executes an ADB command targeting a specific wireless network device."""
    cmd = ["adb", "-s", device_target] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

def orchestrate_recording(device):
    try:
        print(f"[{device}] 🚀 Launching default camera app in video mode...")
        # Broadcast standard system intent to open the native video capture interface
        run_adb(device, ["shell", "am", "start", "-a", "android.media.action.VIDEO_CAPTURE"])
        
        # Critical overhead buffer: Gives the hardware sensor and UI time to initialize
        time.sleep(2.5)  
        
        print(f"[{device}] 🔴 STARTING RECORDING NOW...")
        # Simulates pressing the physical/virtual camera shutter button (KEYCODE_CAMERA)
        run_adb(device, ["shell", "input", "keyevent", "27"])
        
        # Keep this specific thread suspended while the phone records over the air
        time.sleep(RECORD_DURATION)
        
        print(f"[{device}] ⏹️ STOPPING RECORDING...")
        # Press the shutter button again to stop recording and compile the video
        run_adb(device, ["shell", "input", "keyevent", "27"])
        
        # Critical flush buffer: Gives the Android OS time to finish writing the file to storage
        time.sleep(2.5)  
        
        print(f"[{device}] 🔍 Locating the compiled video file on local storage...")
        # List files in the standard Android camera folder, sorted by newest first (-t)
        files_raw = run_adb(device, ["shell", "ls", "-t", "/sdcard/DCIM/Camera/"])
        mp4_files = [f.strip() for f in files_raw.splitlines() if f.strip().lower().endswith(".mp4")]
        
        if not mp4_files:
            print(f"[{device}] ❌ Error: Could not find any recorded .mp4 file in /DCIM/Camera/")
            return
            
        latest_file = mp4_files[0]
        remote_path = f"/sdcard/DCIM/Camera/{latest_file}"
        
        # Format the filename cleanly for Windows filesystem (removes dots and colons from IP)
        sanitized_device_name = device.replace(".", "_").replace(":", "_")
        local_filename = f"{sanitized_device_name}_{latest_file}"
        local_path = os.path.join(LOCAL_SAVE_DIR, local_filename)
        
        print(f"[{device}] 📥 Pulling video over Wi-Fi: {latest_file} -> Windows...")
        run_adb(device, ["pull", remote_path, local_path])
        print(f"[{device}] 🎉 Video successfully transferred!")
        
        # Post-execution cleanup: Force close the camera app so the phone rests cleanly
        run_adb(device, ["shell", "am", "force-stop", "com.android.camera"])
        
    except Exception as e:
        print(f"[{device}] ❌ Automation pipeline crashed: {e}")

def main():
    # Ensure the target storage folder exists on Windows
    if not os.path.exists(LOCAL_SAVE_DIR):
        os.makedirs(LOCAL_SAVE_DIR)
        
    print(f"Initializing synchronization engine for {len(DEVICES)} wireless device(s)...")
    print(f"Videos will be pulled directly to: {LOCAL_SAVE_DIR}\n" + "-"*50)
    
    threads = []
    # Utilize Python threads to dispatch the record command to all phones simultaneously
    for device in DEVICES:
        t = threading.Thread(target=orchestrate_recording, args=(device,))
        threads.append(t)
        t.start()
        
    # Wait for all background download threads to finish safely before closing down the terminal
    for t in threads:
        t.join()
        
    print("\n[COMPLETE] All active pipelines closed down cleanly. Check C:\\PhoneRecordings")

if __name__ == "__main__":
    main()