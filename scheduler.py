import time
import subprocess
import threading

def run_worker():
    while True:
        print("Running AQI Pipeline...")
        subprocess.run(["python", "aqi_system.py"])
        print("Sleeping for 1 hour...")
        time.sleep(3600)

t = threading.Thread(target=run_worker, daemon=True)
t.start()
