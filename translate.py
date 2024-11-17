# translate.py

from googletrans import Translator
import os

# Directories
INPUT_DIR = 'Transcripts_selected'  # Directory containing Kannada .txt files
OUTPUT_DIR = 'Transcripts-English'  # Directory to save translated English .txt files

# Initialize Google Translator
translator = Translator()

def translate_text(text, src_lang='kn', dest_lang='en'):
    """
    Translates text from src_lang to dest_lang using googletrans.
    """
    translation = translator.translate(text, src=src_lang, dest=dest_lang)
    return translation.text

def translate_file(input_file, output_file):
    """
    Translates the content of input_file from Kannada to English and saves it to output_file.
    """
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    translated_lines = []
    for line in lines:
        if line.strip():  # Only translate non-empty lines
            translated_line = translate_text(line)
            translated_lines.append(translated_line)
        else:
            translated_lines.append(line)  # Preserve empty lines

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(translated_lines)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    if not files:
        print(f"No .txt files found in the input directory: {INPUT_DIR}")
        return

    for filename in files:
        input_file = os.path.join(INPUT_DIR, filename)
        output_file = os.path.join(OUTPUT_DIR, filename)
        print(f"Translating {input_file} to {output_file}...")
        translate_file(input_file, output_file)
        print(f"Translation of {input_file} completed.")

if __name__ == "__main__":
    main()