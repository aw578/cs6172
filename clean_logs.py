import os
import re
from pathlib import Path

# Regex for OSC (Operating System Command) sequences: ESC ] ... BEL or ESC \\ (ST)
osc_seq = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

# Improved prompt regex: start of line, user@host: path$ or #, then optional space, then command
PROMPT_REGEX = re.compile(r"^[^@\s]+@[^:]+:[^$\n]+[$#] ?(.*)")


def is_prompt(line):
    # Remove ANSI escape sequences for prompt detection
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    line_clean = ansi_escape.sub("", line)
    # Typical prompt: username@hostname: path$
    return bool(re.match(r".*@.*:.*[$#] ?$", line_clean.strip()))


def strip_prompt_prefix(line):
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    line_clean = ansi_escape.sub("", line)
    # Remove any prompt-like prefix at the start of the line, even if repeated or concatenated
    # Match: anything up to and including @hostname: path$ (possibly repeated)
    prompt_pattern = re.compile(r"(([^\s@]+@[^\s:]+:[^$\n]+[$#] ?)+)")
    return prompt_pattern.sub("", line_clean, count=1).lstrip()


def extract_command(line):
    # Remove OSC sequences
    line = osc_seq.sub("", line)
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    line_clean = ansi_escape.sub("", line)
    match = PROMPT_REGEX.match(line_clean.strip())
    if match:
        cmd = match.group(1).strip()
        # Remove any backspace characters and their effects
        cmd = re.sub(r".\x08", "", cmd)  # Remove backspace and the character before it
        return cmd
    return None


def clean_log_content(content):
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    lines = content.splitlines()
    cleaned_lines = []
    in_nano = False
    current_command = None
    command_output = []

    for i, line in enumerate(lines):
        # Remove ANSI escape sequences
        line_clean = ansi_escape.sub("", line)

        # Remove the script started line
        if line_clean.startswith("Script started on"):
            continue

        # Handle backspaces and escapes
        line_clean = re.sub(r".\x08", "", line_clean)  # Remove backspace and the character before it
        line_clean = re.sub(r"\x1B\[K", "", line_clean)  # Remove clear line escape sequence

        # Detect nano command
        if not in_nano and re.search(r"\bnano\b", line_clean):
            if current_command:
                cleaned_lines.append(f"user: {current_command}")
                if command_output:
                    cleaned_lines.extend(command_output)
                command_output = []
            current_command = line_clean.strip()
            cleaned_lines.append(f"user: {current_command}")
            cleaned_lines.append("nano output")
            in_nano = True
            continue

        # End nano output when prompt reappears
        if in_nano:
            if is_prompt(line):
                in_nano = False
            else:
                continue  # skip nano output

        # Remove prompt lines
        if is_prompt(line):
            if current_command:
                cleaned_lines.append(f"user: {current_command}")
                if command_output:
                    cleaned_lines.extend(command_output)
                command_output = []
            current_command = None
            continue

        # Remove lines that are empty or only non-printable
        if not line_clean.strip() or not any(c.isprintable() and not c.isspace() for c in line_clean):
            continue

        # Remove remaining non-printable/control characters
        line_clean = re.sub(r"[\x00-\x1F\x7F]", "", line_clean)

        # Strip any prompt-like prefixes from the line
        line_clean = strip_prompt_prefix(line_clean)

        # Skip empty lines after cleaning
        if line_clean.strip():
            if current_command is None:
                current_command = line_clean.strip()
            else:
                command_output.append(line_clean.strip())

    # Handle the last command if there is one
    if current_command:
        cleaned_lines.append(f"user: {current_command}")
        if command_output:
            cleaned_lines.extend(command_output)

    # Join lines and ensure proper line endings
    content = "\n".join(cleaned_lines)
    if content and not content.endswith("\n"):
        content += "\n"
    return content


def process_logs():
    # Create output directory if it doesn't exist
    output_dir = Path("cleaned_logs")
    output_dir.mkdir(exist_ok=True)

    # Process all log files in the unparsed_logs directory
    input_dir = Path("unparsed_logs")
    for log_file in input_dir.glob("*.log"):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()

            cleaned_content = clean_log_content(content)

            # Write cleaned content to new file
            output_file = output_dir / log_file.name
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(cleaned_content)

            print(f"Processed {log_file.name}")

        except Exception as e:
            print(f"Error processing {log_file.name}: {str(e)}")


if __name__ == "__main__":
    process_logs()
