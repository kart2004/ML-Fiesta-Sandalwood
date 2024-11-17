from flask import Flask, render_template, request, redirect, url_for
import os
from audiorecord import record_audio

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record():
    duration = int(request.form['duration'])
    filename = "recorded_audio.wav"
    record_audio(filename, duration)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)