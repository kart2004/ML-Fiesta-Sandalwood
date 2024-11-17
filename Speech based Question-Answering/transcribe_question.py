import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os

def transcribe_audio(wav_file_path):
    # Load model and processor
    print("Loading model and processor...")
    processor = Wav2Vec2Processor.from_pretrained("amoghsgopadi/wav2vec2-large-xlsr-kn")
    model = Wav2Vec2ForCTC.from_pretrained("amoghsgopadi/wav2vec2-large-xlsr-kn")
    print("Model and processor loaded.")

    # Define max_length for truncation
    segment_duration = 30  # Adjust to a manageable length, e.g., 30 seconds
    max_length = segment_duration * 16000  # Assuming 16 kHz sampling rate

    # Try loading the .wav file with librosa
    try:
        speech_array, sampling_rate = librosa.load(wav_file_path, sr=16000)  # Resample to 16 kHz if necessary
    except Exception as e:
        print(f"Error loading audio file {wav_file_path}: {e}")
        return None

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
        segments.append(segment)

    # Process segments
    file_transcripts = []
    for segment in segments:
        input_values = processor(segment, sampling_rate=sampling_rate, return_tensors="pt", padding=True, truncation=True, max_length=max_length).input_values
        
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        file_transcripts.append(transcription)
    
    # Combine all transcriptions into a single string
    full_transcription = "\n".join(file_transcripts)
    
    # Save transcription to file
    transcript_file = os.path.join(os.path.dirname(wav_file_path), "recorded_audio.txt")
    with open(transcript_file, 'w', encoding='utf-8') as f:
        f.write(full_transcription)
    print(f"Transcript saved to {transcript_file}")
    return full_transcription