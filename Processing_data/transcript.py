import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import noisereduce as nr
import os

# Load model and processor
print("Loading model and processor...")
processor = Wav2Vec2Processor.from_pretrained("amoghsgopadi/wav2vec2-large-xlsr-kn")
model = Wav2Vec2ForCTC.from_pretrained("amoghsgopadi/wav2vec2-large-xlsr-kn")
print("Model and processor loaded.")

# Directory containing WAV files
wav_dir = os.path.expanduser("../Dataset")  # Adjusted to the correct path

# Directory to save transcripts
transcripts_dir = os.path.expanduser("../Transcripts_all/Transcripts")
os.makedirs(transcripts_dir, exist_ok=True)
print(f"Transcripts will be saved to: {transcripts_dir}")

# Segment duration in seconds
segment_duration = 30  # Adjust to a manageable length, e.g., 30 seconds
batch_size = 4  # Number of segments to process in a batch

# Define max_length for truncation
max_length = segment_duration * 16000  # Assuming 16 kHz sampling rate

# Apply noise reduction
def reduce_noise(audio, sampling_rate):
    reduced_noise_audio = nr.reduce_noise(y=audio, sr=sampling_rate)
    return reduced_noise_audio

# Detect and remove silent segments
def remove_silence(audio, sampling_rate, top_db=20):
    non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
    non_silent_audio = []
    for start, end in non_silent_intervals:
        non_silent_audio.extend(audio[start:end])
    return np.array(non_silent_audio)

# Generate transcripts
for wav_file in os.listdir(wav_dir):
    if wav_file.endswith(".wav"):
        wav_path = os.path.join(wav_dir, wav_file)
        print(f"Processing file: {wav_path}")

        # Try loading the .wav file with librosa
        try:
            speech_array, sampling_rate = librosa.load(wav_path, sr=16000)  # Resample to 16 kHz if necessary
        except Exception as e:
            print(f"Error loading audio file {wav_file}: {e}")
            continue

        # Apply noise reduction
        speech_array = reduce_noise(speech_array, sampling_rate)

        # Remove silence
        speech_array = remove_silence(speech_array, sampling_rate)

        # Split audio into segments
        segments = [speech_array[i:i + max_length] for i in range(0, len(speech_array), max_length)]

        # Process each segment
        for i, segment in enumerate(segments):
            input_values = processor(segment, sampling_rate=sampling_rate, return_tensors="pt", padding="longest").input_values
            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

            # Save transcription to file
            transcript_file = os.path.join(transcripts_dir, f"{os.path.splitext(wav_file)[0]}_segment_{i}.txt")
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(transcription)
            print(f"Saved transcript segment {i} to {transcript_file}")

print("Transcription process completed.")