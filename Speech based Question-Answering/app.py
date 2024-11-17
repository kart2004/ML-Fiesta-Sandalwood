import os
import subprocess
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, render_template, request, jsonify
from audiorecord import record_audio
from transcribe_question import transcribe_audio

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record():
    duration = int(request.form['duration'])
    filename = "recorded_audio.wav"
    record_audio(filename, duration)
    transcription = transcribe_audio(filename)
    
    # Save the transcription to a file
    transcription_file = "recorded_audio.txt"
    with open(transcription_file, 'w', encoding='utf-8') as f:
        f.write(transcription)
    
    # Print the transcription before running the subprocess
    print(f"Transcription: {transcription}")
    
    # Run the updated_search_segment.py script
    subprocess.run(["python", "updated_search_segment.py"])
    # Run the searchsegpt2.py script
    subprocess.run(["python", "searchsegpt2.py"])
    
    return render_template('index.html', transcription=transcription)

if __name__ == "__main__":
    app.run(debug=True)