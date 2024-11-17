import os
from dotenv import load_dotenv
from groq import Groq
import json
import re
import torchaudio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()  

# Step 2: Read the contents of the file (document content)
def load_file_contents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Step 3: Read the query from a file
def load_query_from_file(query_file_path):
    with open(query_file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()  # Read the query and remove extra spaces/newlines
def find_best_matching_transcript(answer_file, transcripts_dir):
    # Load the contents of the answers file
    answer_content = load_file_contents(answer_file)
    
    # Collect all transcript files and their contents
    transcript_files = [f for f in os.listdir(transcripts_dir) if f.endswith('.txt')]
    transcript_contents = [load_file_contents(os.path.join(transcripts_dir, f)) for f in transcript_files]
    
    # Combine the answer content with transcript contents for TF-IDF vectorization
    all_texts = [answer_content] + transcript_contents
    
    # Compute TF-IDF vectors for all texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Calculate cosine similarity between the answer content and all transcripts
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Find the index of the best matching transcript
    best_match_idx = similarities.argmax()
    
    # Return the file name of the best matching transcript
    return transcript_files[best_match_idx], similarities[best_match_idx]

def get_timestamp_from_file(file_path):
    # Define the regex pattern to match timestamps (e.g., [30s] or [450ms])
    pattern = r'\[(\d+)(ms|s)\]'
    
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Search for the first occurrence of the pattern
    match = re.search(pattern, content)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        # Convert timestamp to seconds
        return value / 1000.0 if unit == "ms" else value
    else:
        raise ValueError("No timestamp found in the file.")

# Step 4: Use the Groq API for chat completion
def ask_groq_api(file_contents, query, api_key):
    # Set up Groq API client
    client = Groq(api_key=api_key)
    
    # Construct the prompt
    prompt = f"search for the exact answer and return the sentence from the summary with the timestamp. like for example, [0s] sentence. Don't change anything, just return the exact sentence\n\n" \
         f"file contents:\n{file_contents}\n\n" \
         f"query:\n{query}\n\n" \
         f"return the answer in kannada only"



    # Prepare the context with the constructed prompt
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,  # The full prompt containing both the document and query in Kannada
            }
        ],
        model="llama3-8b-8192",  # Replace with the model you want to use
    )
    
    # Extract and return the response content (in Kannada)
    return chat_completion.choices[0].message.content

def get_audio_segment(transcript_file, start_time, segment_duration, audio_dir):
    audio_file = os.path.splitext(transcript_file)[0] + ".wav"
    audio_path = os.path.join(audio_dir, audio_file)
    speech_array, sampling_rate = torchaudio.load(audio_path)
    start_sample = int(start_time * sampling_rate)
    end_sample = int((start_time + segment_duration) * sampling_rate)
    answer_segment = speech_array[:, start_sample:end_sample]
    return answer_segment, sampling_rate
# Step 5: Save the answer to a file
def save_answer_to_file(answer, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(answer)

# Main function to load the files, ask the question, and save the answer
def main():
    # File paths
    document_file_path = 'answers/ans.txt'  # Path to your Kannada document file
    query_file_path = 'newtranscripts/recorded_audio.txt'  # Path to your Kannada query file
    output_file_path = 'answers/answerff.txt'  # Path to save the answer in Kannada
    
    # Get the API key from environment variable
    api_key = os.getenv("GROQ_API_KEY")  # Retrieve the API key from the .env file

    if not api_key:
        print("Error: GROQ_API_KEY is not set in environment variables.")
        return

    # Step 1: Load the file contents (document in Kannada)
    file_contents = load_file_contents(document_file_path)
    
    # Step 2: Load the query from a file (in Kannada)
    query = load_query_from_file(query_file_path)
    
    # Step 3: Ask the Groq API with the document and query (both in Kannada)
    answer = ask_groq_api(file_contents, query, api_key)
    
    # Step 4: Save the answer to a file (in Kannada)
    save_answer_to_file(answer, output_file_path)
    answer_file_path = 'answers/ans.txt'
    transcripts_dir = 'transcripts3'
    
    # Find the best matching transcript file
    best_match_file, similarity_score = find_best_matching_transcript(answer_file_path, transcripts_dir)
    
    print(f"Best matching transcript file: {best_match_file}")
    print(f"Similarity score: {similarity_score}")
    print(f"Answer saved to {output_file_path}")

    path = "answers/answerfinal.txt"
    start_time = get_timestamp_from_file(path)
    segment_duration = 30  # Adjust as needed
    audio_dir = "Dataset-wav"
    answer_segment, sampling_rate = get_audio_segment(best_match_file, start_time, segment_duration, audio_dir)
    answer_audio_path = "answers/answer_segment.wav"
    torchaudio.save(answer_audio_path, answer_segment, sampling_rate)
    print(f"Answer audio segment saved as {answer_audio_path}")

if __name__ == "__main__":
    main()
