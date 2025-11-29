import time
from datetime import datetime, timedelta
import subprocess

print(" Scheduler started — Running AQI pipeline every EXACT 1 hour")

while True:
    # 1. Capture run start time
    run_start = datetime.now()
    print(f"\n Starting AQI cycle at: {run_start.strftime('%Y-%m-%d %H:%M:%S')}")

    # 2. Run pipeline
    try:
        result = subprocess.run(
            ["python3", "aqi_system.py"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("ERROR:", result.stderr)
    except Exception as e:
        print("Pipeline crashed:", e)

    # 3. Calculate next exact run
    run_end = datetime.now()
    next_run = run_start + timedelta(hours=1)

    sleep_seconds = max(0, (next_run - run_end).total_seconds())

    # Log remaining wait time
    mins = int(sleep_seconds // 60)
    secs = int(sleep_seconds % 60)
    print(f" Next run at: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Sleeping for {mins} minutes {secs} seconds...")

    # 4. Sleep UNTIL next run
    time.sleep(sleep_seconds)
