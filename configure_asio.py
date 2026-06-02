import os
import time

# Turn on the hidden ASIO switch
os.environ["SD_ENABLE_ASIO"] = "1"

# NOW import sounddevice
import sounddevice as sd

# Check the APIs again
apis = [api['name'] for api in sd.query_hostapis()]
print("Available APIs:", apis)

try:
    asio_id = next(i for i, d in enumerate(sd.query_devices()) if "ASIO4ALL" in d['name'])
    print("Found ASIO4ALL. Opening stream to trigger the control panel...")

    # Open a dummy stream to force the system tray icon to appear
    with sd.InputStream(device=asio_id, channels=1, samplerate=48000):
        time_awake = 120
        print("Look at your Windows System Tray (bottom right) for a green play button icon with an arrow.")
        print(f"You have {time_awake} seconds to open it...")
        time.sleep(time_awake)
except StopIteration:
    print("Could not find ASIO4ALL. Did you install it, or do you need to restart your IDE?")