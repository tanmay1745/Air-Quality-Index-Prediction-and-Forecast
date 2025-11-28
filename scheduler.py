import threading
import time
import subprocess

def run_pipeline_loop():
    while True:
        print("Running AQI pipeline...")
        subprocess.run(["python", "aqi_system.py"])
        print("Pipeline completed. Sleeping 1 hour...")
        time.sleep(3600)

# Start background thread
thread = threading.Thread(target=run_pipeline_loop, daemon=True)
thread.start()
