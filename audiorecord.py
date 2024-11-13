import speech_recognition as sr

def record_audio(filename="recorded_audio.wav", duration=10):
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print(f"Recording for {duration} seconds... Please speak.")
        
        # Adjust for ambient noise and listen for audio
        recognizer.adjust_for_ambient_noise(source)
        
        # Record the audio for the specified duration
        audio = recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
        
        # Save the recorded audio as a WAV file
        with open(filename, "wb") as f:
            f.write(audio.get_wav_data())
        
        print(f"Audio saved as {filename}")

# Record audio and save it to "recorded_audio.wav" for 10 seconds
record_audio("recorded_audio.wav")
