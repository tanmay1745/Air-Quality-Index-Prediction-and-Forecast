import time
import subprocess

while True:
    print("Running hourly AQI pipeline...")
    subprocess.run(["python", "aqi_system.py"])
    print("Done. Sleeping for 1 hour.")
    time.sleep(3600)
