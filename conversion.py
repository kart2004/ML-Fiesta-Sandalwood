import os
from pydub import AudioSegment

# Directory containing MP3 files
mp3_dir = r"ML-Fiesta-Sandalwood/Dataset"
wav_dir = r"ML-Fiesta-Sandalwood/Dataset-wav"

# Ensure the WAV directory exists
os.makedirs(wav_dir, exist_ok=True)

# Convert MP3 to WAV
for mp3_file in os.listdir(mp3_dir):
    if mp3_file.endswith(".mp3"):
        mp3_path = os.path.join(mp3_dir, mp3_file)
        wav_path = os.path.join(wav_dir, os.path.splitext(mp3_file)[0] + ".wav")
        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio.set_frame_rate(16000).set_channels(1)  # Ensure the audio is 16 kHz and mono
        audio.export(wav_path, format="wav")