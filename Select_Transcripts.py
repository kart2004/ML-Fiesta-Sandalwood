# select_transcripts.py

import os
import re
import logging
from tqdm import tqdm

# Define the input and output directories
INPUT_DIR = 'Transcripts'
OUTPUT_DIR = 'Transcripts_selected'

# Define the keywords to search for (including partial matches)
KEYWORDS = ['ಶ್ರೀಗ', 'ಚಂ', 'ಶ್ರೀಗಂಧದ', 'ಚಂದನ']

# Configure logging
logging.basicConfig(
    filename='select_transcripts.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def is_sandalwood_related(text):
    """
    Checks if the text contains any sandalwood-related keywords.
    """
    return any(keyword in text for keyword in KEYWORDS)

def mark_timestamp(line):
    """
    Marks the timestamp in the line with [sandalwood] if it contains related keywords.
    """
    # Regex to match the timestamp and the text
    match = re.match(r'^(\[\d+s\])\s+(.*)', line)
    if match:
        timestamp, text = match.groups()
        # Append [sandalwood] after the timestamp
        return f"{timestamp} [sandalwood] {text}\n"
    else:
        # If the line doesn't match the expected format, return it as is
        return line if line.strip() else ""

def process_file(filename):
    """
    Processes a single transcript file:
    - Checks if the file contains any sandalwood-related keywords.
    - If yes, marks the corresponding timestamps with [sandalwood].
    - Writes the entire (modified) transcript to the output directory.
    """
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
    except Exception as e:
        logging.error(f"Error reading {input_path}: {e}")
        return

    # Check if any line contains the keywords
    contains_keyword = any(is_sandalwood_related(line) for line in lines)

    if not contains_keyword:
        # If no keywords found, skip this file
        logging.info(f"No sandalwood keywords found in: {filename}. Skipping.")
        return

    # Process all lines, marking those that contain keywords
    processed_lines = []
    for line in lines:
        if is_sandalwood_related(line):
            marked_line = mark_timestamp(line)
            processed_lines.append(marked_line)
        else:
            processed_lines.append(line if line.strip() else "")

    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.writelines(processed_lines)
        print(f"Selected and marked: {filename}")
        logging.info(f"Successfully processed and selected: {filename}")
    except Exception as e:
        logging.error(f"Error writing to {output_path}: {e}")

def select_transcripts():
    """
    Iterates through all transcript files in the input directory
    and processes them to create selected transcripts with marked timestamps.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created output directory: {OUTPUT_DIR}")

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]

    if not files:
        logging.warning(f"No .txt files found in the input directory: {INPUT_DIR}")
        print(f"No .txt files found in the input directory: {INPUT_DIR}")
        return

    for filename in tqdm(files, desc="Processing Transcripts"):
        process_file(filename)

if __name__ == "__main__":
    select_transcripts()