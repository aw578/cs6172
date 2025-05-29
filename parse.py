# Please write a python script that calls replay.py on all the files in unparsed_logs/ and saves the output in parsed_logs/.
# example: python replay.py ./unparsed_logs/2025-05-10-17-01-00.log 206 10000 > ./parsed_logs/2025-05-10-17-01-00.log
import os
import subprocess
from pathlib import Path

# Create parsed_logs directory if it doesn't exist
Path("parsed_logs").mkdir(exist_ok=True)

# Process each log file in unparsed_logs
for log_file in Path("unparsed_logs").glob("*.log"):
    output_file = Path("parsed_logs") / log_file.name

    # Run replay.py with appropriate dimensions
    # Using 206 columns and 10000 lines as shown in the example
    cmd = ["python", "replay.py", str(log_file), "206", "100000"]

    # Run the command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Write output to parsed_logs directory
    with open(output_file, "w") as f:
        f.write(result.stdout)
