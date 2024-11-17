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
wav_dir = os.path.expanduser("Dataset-wav") 

# Directory to save transcripts
transcripts_dir = os.path.expanduser("Transcripts")
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

        # Save the transcripts for the current file
        transcript_file = os.path.join(transcripts_dir, os.path.splitext(wav_file)[0] + ".txt")
        print(f"Saving transcript to: {transcript_file}")
        with open(transcript_file, "w", encoding="utf-8") as f:
            for transcript in file_transcripts:
                f.write(f"[{transcript['start_time']}s]\t{transcript['transcription']}\n")
        print(f"Transcript for {wav_file} saved.")