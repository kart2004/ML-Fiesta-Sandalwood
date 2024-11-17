import os
from transformers import BertTokenizer, BertModel
import torch
import faiss
import numpy as np

# Path to the folder containing transcripts
transcripts_folder = 'transcripts3'
def load_question_from_file(question_file):
    with open(question_file, 'r', encoding='utf-8') as file:
        return file.read().strip()  # Read the question and remove extra spaces/newlines

# Load a pre-trained multilingual model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Function to read the content of all transcript files
def read_transcripts(folder_path):
    transcripts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Assuming transcripts are in text files
            with open(os.path.join(folder_path, filename), encoding='utf-8') as file:
                content = file.read()
                transcripts.append(content)
    return transcripts

# Read the transcripts from the folder
transcripts = read_transcripts(transcripts_folder)

# Function to generate embeddings using mBERT
def generate_embeddings(texts):
    embeddings = []
    for text in texts:
        # Tokenize the text
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        # Get the model's output (hidden states)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the mean of the last hidden state as the embedding
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)
    
    return np.array(embeddings)

# Generate embeddings for all transcripts
embeddings = generate_embeddings(transcripts)

# Convert embeddings to numpy array (FAISS requires numpy float32 format)
embedding_matrix = embeddings.astype(np.float32)

# Create a FAISS index (using L2 distance)
index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # L2 distance (Euclidean)

# Add embeddings to the FAISS index
index.add(embedding_matrix)

# Save the FAISS index to a file (optional, for persistent storage)
faiss.write_index(index, "transcript_in.faiss")

# Function to vectorize and search for the answer to a Kannada question
def get_answer_from_transcripts(question):
    # Vectorize the user's question
    inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    # Get the model's output (hidden states)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the mean of the last hidden state as the embedding
        question_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    # Convert question embedding to numpy array for FAISS search
    query_vector = np.array(question_embedding).astype(np.float32).reshape(1, -1)
    
    # Perform search for the top 3 most relevant segments
    k = 1
    distances, indices = index.search(query_vector, k)
    
    # Return the top 3 closest transcript segments
    answers = []
    for i in range(k):
        answers.append(f"Segment: {transcripts[indices[0][i]]}\nDistance: {distances[0][i]}")
    
    return answers

# Example usage
# Function to save answers to a text file
def save_answers_to_file(answers, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for answer in answers:
            file.write(answer + '\n\n')  # Separate answers with a blank line

# Example usage
question_file = "recorded_audio.txt"
question = load_question_from_file(question_file)
# Kannada question
answers = get_answer_from_transcripts(question)

# Save the answers to a text file
output_file = "answers/ans.txt"
save_answers_to_file(answers, output_file)

print(f"Answers saved to {output_file}")

