import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os

# Load model and processor
print("Loading model and processor...")
processor = Wav2Vec2Processor.from_pretrained("amoghsgopadi/wav2vec2-large-xlsr-kn")
model = Wav2Vec2ForCTC.from_pretrained("amoghsgopadi/wav2vec2-large-xlsr-kn")
print("Model and processor loaded.")

# Path to the single WAV file
wav_file_path = "recorded_audio.wav"  # Replace with the actual path

# Directory to save transcript
transcripts_dir = "newtranscripts"
os.makedirs(transcripts_dir, exist_ok=True)
print(f"Transcript will be saved to: {transcripts_dir}")

# Segment duration in seconds
segment_duration = 30  # Adjust to a manageable length, e.g., 30 seconds
batch_size = 4  # Number of segments to process in a batch

# Define max_length for truncation
max_length = segment_duration * 16000  # Assuming 16 kHz sampling rate

# Try loading the .wav file with librosa
try:
    speech_array, sampling_rate = librosa.load(wav_file_path, sr=16000)  # Resample to 16 kHz if necessary
except Exception as e:
    print(f"Error loading audio file {wav_file_path}: {e}")
    exit(1)

# Total length of audio in seconds
total_duration = len(speech_array) / sampling_rate
print(f"Total duration of audio: {total_duration:.2f} seconds")

# Split into segments
segments = []
for i in range(0, int(total_duration), segment_duration):
    # Calculate start and end in sample points
    start_sample = i * sampling_rate
    end_sample = min((i + segment_duration) * sampling_rate, len(speech_array))
    segment = speech_array[start_sample:end_sample]
    segments.append((segment, i))

# Process segments in batches
file_transcripts = []
for batch_start in range(0, len(segments), batch_size):
    batch_segments = segments[batch_start:batch_start + batch_size]
    print(f"Processing batch from segment {batch_start} to {batch_start + len(batch_segments) - 1}")
    batch_inputs = [processor(segment[0], sampling_rate=sampling_rate, return_tensors="pt", padding=True, truncation=True, max_length=max_length) for segment in batch_segments]
    
    # Ensure all input tensors have the same length by padding
    input_values = torch.nn.utils.rnn.pad_sequence([inputs.input_values.squeeze(0) for inputs in batch_inputs], batch_first=True)
    attention_mask = torch.nn.utils.rnn.pad_sequence([inputs.attention_mask.squeeze(0) for inputs in batch_inputs], batch_first=True)
    
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcriptions = processor.batch_decode(predicted_ids)
    
    for (segment, start_time), transcription in zip(batch_segments, transcriptions):
        file_transcripts.append({"start_time": start_time, "transcription": transcription})
        print(f"Transcription of segment starting at {start_time} seconds: {transcription}")

# Save the transcript for the current file
transcript_file = os.path.join(transcripts_dir, os.path.splitext(os.path.basename(wav_file_path))[0] + ".txt")
print(f"Saving transcript to: {transcript_file}")
with open(transcript_file, "w", encoding="utf-8") as f:
    for transcript in file_transcripts:
        f.write(f"[{transcript['start_time']}s]\t{transcript['transcription']}\n")
print(f"Transcript for {wav_file_path} saved.")