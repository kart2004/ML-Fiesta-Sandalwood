# select_transcripts.py

import os
import re
import logging
from tqdm import tqdm

# Define the input and output directories
transcripts_dir = r"../Transcripts_all/Transcripts"
selected_transcripts_dir = r"../Transcripts_all/Transcripts_selected"

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
    input_path = os.path.join(transcripts_dir, filename)
    output_path = os.path.join(selected_transcripts_dir, filename)

    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
    except Exception as e:
        logging.error(f"Error reading {input_path}: {e}")
        return

    contains_keywords = any(is_sandalwood_related(line) for line in lines)

    if contains_keywords:
        try:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                for line in lines:
                    if is_sandalwood_related(line):
                        outfile.write(mark_timestamp(line))
                    else:
                        outfile.write(line)
            logging.info(f"Processed and saved: {output_path}")
        except Exception as e:
            logging.error(f"Error writing {output_path}: {e}")

def main():
    if not os.path.exists(selected_transcripts_dir):
        os.makedirs(selected_transcripts_dir)

    files = [f for f in os.listdir(transcripts_dir) if f.endswith('.txt')]
    if not files:
        print(f"No .txt files found in the input directory: {transcripts_dir}")
        return

    for filename in tqdm(files, desc="Processing Files"):
        process_file(filename)

if __name__ == "__main__":
    main()