import os
import torchaudio
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the user's question from 'recorded_audio.txt'
def load_user_question(file_path="newtranscripts/recorded_audio.txt"):
    with open(file_path, "r", encoding="utf-8") as file:
        question = file.read().strip()
    return question

# Step 2: Load all transcript files from the transcripts directory
def load_transcripts(transcripts_dir="Transcripts"):
    transcripts = []
    files = os.listdir(transcripts_dir)
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(transcripts_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
                transcripts.append((file, transcript))  # Store file name and content
    return transcripts

# Step 3: Encode text using Sentence-Transformer to get embeddings
def encode_texts(texts, model):
    return model.encode(texts, convert_to_tensor=True)

# Step 4: Find the most relevant transcript using cosine similarity on embeddings
def find_best_match(question, transcripts, model):
    # Encode the question
    question_embedding = model.encode([question], convert_to_tensor=True)

    # Encode the transcripts
    transcript_texts = [transcript[1] for transcript in transcripts]
    transcript_embeddings = model.encode(transcript_texts, convert_to_tensor=True)

    # Move tensors to CPU before calculating cosine similarity
    question_embedding_cpu = question_embedding.cpu().detach().numpy()
    transcript_embeddings_cpu = transcript_embeddings.cpu().detach().numpy()

    # Compute cosine similarity between the question and each transcript
    cosine_similarities = cosine_similarity(question_embedding_cpu, transcript_embeddings_cpu)

    # Find the index of the most similar transcript
    best_match_idx = cosine_similarities.argmax()

    # Return the file name and content of the best matching transcript
    return transcripts[best_match_idx]

# Step 5: Get the audio segment corresponding to the best match
def get_audio_segment(transcript_file, start_time, segment_duration, audio_dir):
    audio_file = os.path.splitext(transcript_file)[0] + ".wav"
    audio_path = os.path.join(audio_dir, audio_file)
    speech_array, sampling_rate = torchaudio.load(audio_path)
    start_sample = int(start_time * sampling_rate)
    end_sample = int((start_time + segment_duration) * sampling_rate)
    answer_segment = speech_array[:, start_sample:end_sample]
    return answer_segment, sampling_rate

# Step 6: Main function to run the pipeline
def qa_pipeline():
    # Load the user's question
    question = load_user_question("newtranscripts/recorded_audio.txt")

    # Load transcripts from the directory
    transcripts = load_transcripts("Transcripts")

    # Load the pre-trained Sentence-Transformer model (mBERT or XLM-R)
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    # Find the best match
    best_match_file, best_match_transcript = find_best_match(question, transcripts, model)

    # Output the result
    print(f"Best matching transcript found in file: {best_match_file}")
    print(f"Answer: {best_match_transcript}")

    # Ensure the answers directory exists
    os.makedirs("answers", exist_ok=True)

    # Save the answer to a file
    output_file = "answers/answer.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Question: {question}\n")
        f.write(f"Best matching transcript found in file: {best_match_file}\n")
        f.write(f"Answer: {best_match_transcript}\n")

    print(f"Answer has been saved to {output_file}")

    # Save the corresponding audio segment
    start_time = float(best_match_transcript.split("[")[1].split("s]")[0])
    segment_duration = 30  # Adjust as needed
    audio_dir = "Dataset-wav"
    answer_segment, sampling_rate = get_audio_segment(best_match_file, start_time, segment_duration, audio_dir)
    answer_audio_path = "answers/answer_segment.wav"
    torchaudio.save(answer_audio_path, answer_segment, sampling_rate)
    print(f"Answer audio segment saved as {answer_audio_path}")

# Run the QA pipeline
qa_pipeline()