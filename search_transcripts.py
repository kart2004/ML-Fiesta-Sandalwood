import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the user's question from 'recordedaudio.txt'
def load_user_question(file_path="newtranscripts/recorded_audio.txt"):
    with open(file_path, "r", encoding="utf-8") as file:
        question = file.read().strip()
    return question

# Step 2: Load all transcript files from the transcripts directory
def load_transcripts(transcripts_dir="newstuff"):
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

# Step 5: Main function to run the pipeline
def qa_pipeline():
    # Load the user's question
    question = load_user_question("newtranscripts/recorded_audio.txt")

    # Load transcripts from the directory
    transcripts = load_transcripts("newstuff")

    # Load the pre-trained Sentence-Transformer model (mBERT or XLM-R)
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    # Find the best match
    best_match_file, best_match_transcript = find_best_match(question, transcripts, model)

    # Output the result
    print(f"Best matching transcript found in file: {best_match_file}")
    print(f"Answer: {best_match_transcript}")

    # Save the answer to a file
    output_file = "answers/answer.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Question: {question}\n")
        f.write(f"Best matching transcript found in file: {best_match_file}\n")
        f.write(f"Answer: {best_match_transcript}\n")

    print(f"Answer has been saved to {output_file}")

# Run the QA pipeline
qa_pipeline()