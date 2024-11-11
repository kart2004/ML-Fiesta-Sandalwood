from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa

# Convert MP3 to WAV
#mp3_audio = AudioSegment.from_mp3(r"Dataset\SandalWoodNewsStories_1.mp3")
#mp3_audio.export(r"Dataset\SandalWoodNewsStories_1.wav", format="wav")

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="kannada", task="transcribe")

# load your own Kannada audio file
audio_path = r"hah.wav"
audio, sr = librosa.load(audio_path, sr=16000)

inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
input_features = inputs.input_features
attention_mask = inputs.attention_mask

# generate token ids
predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription)