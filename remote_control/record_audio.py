import subprocess
import time
import threading
import os
import shutil
import re
from moviepy import VideoFileClip
from zeroconf import Zeroconf, ServiceBrowser
from datetime import datetime

# ==========================================
#     PERMANENT HARDWARE ALIAS REGISTRY
# ==========================================
# Maps the phone's factory model number to your custom alias.
HARDWARE_REGISTRY = {
    "sms926b": "Amibar",     # Samsung Galaxy S24
    "sma566b": "Bronfonfon", # Samsung Galaxy A56
    "sma505f": "Backup_Bronfonfon" # Samsung Galaxy A50
}
MIC_INDEX_MAP = {
    "Amibar": "1",
    "Bronfonfon": "2",
    "Backup_Bronfonfon": "3"
}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "video_recordings"))
AUDIO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "audio_recordings"))

# ==========================================
#        AUTOMATIC M-DNS DISCOVERY
# ==========================================
class ADBDeviceListener:
    def __init__(self):
        self.found_services = []

    def update_service(self, zc, type_, name): pass
    def remove_service(self, zc, type_, name): pass

    def add_service(self, zc, type_, name):
        info = zc.get_service_info(type_, name)
        if info:
            self.found_services.append(info)

def discover_live_network_nodes(scan_timeout=3.0):
    """
    Sweeps Wi-Fi for devices, connects, and reads their permanent factory 
    model numbers to bypass all Wi-Fi network abstraction issues.
    """
    print("[Discovery Engine] Resolving active sensor links by hardware model...")
    live_alias_map = {}
    adb_path = get_adb_path()
    
    # Sweep local network to ensure all phones are bridged
    zeroconf = Zeroconf()
    listener = ADBDeviceListener()
    browser = ServiceBrowser(zeroconf, "_adb._tcp.local.", listener)
    browser_tls = ServiceBrowser(zeroconf, "_adb-tls-connect._tcp.local.", listener)
    time.sleep(scan_timeout)
    zeroconf.close()
    
    for service_info in listener.found_services:
        if service_info.parsed_addresses():
            ip = service_info.parsed_addresses()[0]
            target_endpoint = f"{ip}:{service_info.port}"
            subprocess.run([adb_path, "connect", target_endpoint], capture_output=True)

    # Now that everything is connected, ask the ADB daemon for the device list
    result = subprocess.run([adb_path, "devices"], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')[1:] 
    
    for line in lines:
        if not line.strip() or "offline" in line or "unauthorized" in line:
            continue
            
        # Extract the endpoint (e.g., 10.40.206.172:5555 or adb-R5C...)
        target_endpoint = line.split()[0]
        
        # Ask the phone's internal OS for its factory model name
        model_cmd = subprocess.run([adb_path, "-s", target_endpoint, "shell", "getprop", "ro.product.model"], capture_output=True, text=True)
        raw_model = model_cmd.stdout.strip().lower()
        
        # Sanitize the model name (e.g., 'sm-s926b' or 'sm_s926b' becomes 'sms926b')
        clean_model = re.sub(r"[^a-z0-9]", "", raw_model)
        
        if clean_model in HARDWARE_REGISTRY:
            alias = HARDWARE_REGISTRY[clean_model]
            if alias not in live_alias_map:
                live_alias_map[alias] = target_endpoint
                print(f" -> [MATCH SUCCESS] Bound alias '{alias}' to device model '{clean_model}' at {target_endpoint}")
        else:
            print(f" -> [SKIPPED] Connected device model '{clean_model}' not found in registry.")

    return live_alias_map

# ==========================================
#         PIPELINE OPERATIONS
# ==========================================
def get_adb_path():
    adb_lookup = shutil.which("adb")
    if adb_lookup: return adb_lookup
    standard_path = r"C:\platform-tools\adb.exe"
    if os.path.exists(standard_path): return standard_path
    return "adb"

def run_adb(device_target, args):
    cmd = [get_adb_path(), "-s", device_target] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

def extract_and_verify_audio(video_path, audio_path, alias):
    try:
        video = VideoFileClip(video_path)
        src_fps = int(video.audio.fps) if video.audio else 44100  
        video.audio.write_audiofile(audio_path, fps=src_fps, nbytes=2, codec='pcm_s16le', logger=None)
        video.close()
        if os.path.exists(video_path): os.remove(video_path)
    except Exception as e:
        print(f"[{alias}] Local Extraction Error: {e}")


# ==========================================
#   UPGRADED PIPELINE WITH MIC INDEX NAMES
# ==========================================
def orchestrate_pipeline(device_ip, alias, duration, sync_barrier, master_timestamp):
    try:
        run_adb(device_ip, ["shell", "input", "keyevent", "KEYCODE_WAKEUP"])
        time.sleep(0.5)
        
        if alias == "Backup_Bronfonfon":
            print(f"[{alias}] Legacy device detected. Cold-booting main camera activity...")
            run_adb(device_ip, ["shell", "am", "start", "-n", "com.sec.android.app.camera/com.sec.android.app.camera.Camera"])
            time.sleep(3.5)
            
            print(f"[{alias}] Applying legacy UI swipe to force Video mode...")
            run_adb(device_ip, ["shell", "input", "swipe", "800", "1000", "200", "1000", "300"])
            time.sleep(1.5)
        else:
            print(f"[{alias}] Modern device detected. Driving high-level Video Capture Intent...")
            run_adb(device_ip, ["shell", "am", "start", "-a", "android.media.action.VIDEO_CAPTURE", "-p", "com.sec.android.app.camera"])
            time.sleep(3.5)
        
        print(f"[{alias}] Waiting at synchronization barrier for other sensors...")
        sync_barrier.wait(timeout=15.0)
        
        print(f"[{alias}] 🔴 STARTING AUDIO CAPTURE ({duration} seconds)...")
        run_adb(device_ip, ["shell", "input", "keyevent", "25"])
        
        time.sleep(duration)
        
        sync_barrier.wait(timeout=15.0)
        
        print(f"[{alias}] ⏹️ STOPPING AUDIO CAPTURE...")
        run_adb(device_ip, ["shell", "input", "keyevent", "25"])
        
        time.sleep(3.0)  
        
        files_raw = run_adb(device_ip, ["shell", "ls", "-t", "/sdcard/DCIM/Camera/"])
        mp4_files = [f.strip() for f in files_raw.splitlines() if f.strip().lower().endswith(".mp4")]
        
        if not mp4_files:
            print(f"[{alias}] Error: No audio container file found.")
            return
            
        remote_path = f"/sdcard/DCIM/Camera/{mp4_files[0]}"
        
        # Look up the permanent mic index (fallback to alias name if not mapped)
        mic_id = MIC_INDEX_MAP.get(alias, alias)
        
        # New clean file naming format: YYYYMMDD_HHMMSS_micX.wav
        file_base_name = f"{master_timestamp}_mic{mic_id}"
        local_video_path = os.path.join(VIDEO_DIR, f"{file_base_name}.mp4")
        local_audio_path = os.path.join(AUDIO_DIR, f"{file_base_name}.wav")
        
        run_adb(device_ip, ["pull", remote_path, local_video_path])
        run_adb(device_ip, ["shell", "rm", remote_path])
        
        extract_and_verify_audio(local_video_path, local_audio_path, alias)
        run_adb(device_ip, ["shell", "am", "force-stop", "com.sec.android.app.camera"])
        print(f"[{alias}] 🎉 Asset pipeline execution complete.")
        
    except threading.BrokenBarrierError:
        print(f"[{alias}] Sync barrier broke. Aborting.")
    except Exception as e:
        print(f"[{alias}] Pipeline exception error context: {e}")

# ==========================================
#         DYNAMIC MASTER WRAPPER
# ==========================================
def capture_by_alias(devices=None):
    for directory in [VIDEO_DIR, AUDIO_DIR]:
        if not os.path.exists(directory): 
            os.makedirs(directory)
            
    if not devices:
        return
        
    live_network_map = discover_live_network_nodes()
    active_devices = [(alias, length) for alias, length in devices if alias in live_network_map]
    
    if not active_devices:
        print("\n[API Error] No requested devices were found online.")
        return
    
    if len(active_devices) < 2:
        print(f"\n[API Error] Only {len(active_devices)} device(s) detected. Minimum 2 devices required for synchronized recording. Aborting.")
        return

    sync_barrier = threading.Barrier(len(active_devices))
    
    # Generate the single snapshot timestamp accurate to the second
    # Format outcome example: 20260609_165032
    master_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    threads = []
    print(f"\nProcessing execution instructions for {len(active_devices)} online sensor(s)...")
    
    for alias, length in active_devices:
        device_ip = live_network_map[alias]
        # Pass the master_timestamp into every worker thread context
        t = threading.Thread(
            target=orchestrate_pipeline, 
            args=(device_ip, alias, length, sync_barrier, master_timestamp)
        )
        threads.append(t)
        t.start()
            
    for t in threads: 
        t.join()
        
    print(f"\n[COMPLETE] Script operations successfully wrapped up.")


def record_audio(duration, mic_indexes):
    """
    Wrapper around capture_by_alias.

    Args:
        duration: Recording duration in seconds for each selected mic.
        mic_indexes: Iterable of mic indexes (0, 1, 2) to record.
    """
    if duration is None or duration <= 0:
        raise ValueError("duration must be a positive number of seconds")

    index_to_alias = {value: alias for alias, value in MIC_INDEX_MAP.items()}
    requested_devices = []

    for raw_index in mic_indexes:
        index = str(raw_index)
        alias = index_to_alias.get(index)
        if alias is None:
            raise ValueError(f"Unknown mic index: {raw_index}")
        requested_devices.append((alias, duration))

    if not requested_devices:
        raise ValueError("mic_indexes must contain at least one valid mic index")

    capture_by_alias(devices=requested_devices)


if __name__ == "__main__":
    record_audio(duration=30, mic_indexes=[1, 2, 3])
    # discover_live_network_nodes()
    
