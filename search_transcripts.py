# search_transcripts.py

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from googletrans import Translator

# Configuration
LLM_MODEL_NAME = 'EleutherAI/gpt-neo-1.3B'  # More capable LLM that can run on CPU


DEVICE = 0 if torch.cuda.is_available() else -1  # Use GPU if available

# Initialize QA Pipeline with GPT-Neo 1.3B
qa_pipeline = pipeline(
    "text-generation",
    model=LLM_MODEL_NAME,
    tokenizer=LLM_MODEL_NAME,
    device=DEVICE
)

# Initialize Google Translator
translator = Translator()

def translate_kannada_to_english(text):
    """
    Translates text from Kannada to English using googletrans.
    """
    translation = translator.translate(text, src='kn', dest='en')
    return translation.text

def translate_english_to_kannada(text):
    """
    Translates text from English to Kannada using googletrans.
    """
    translation = translator.translate(text, src='en', dest='kn')
    return translation.text

def generate_answer(question):
    """
    Generates an answer in English using the QA pipeline.
    """
    prompt = f"Q: {question}\nA:"
    response = qa_pipeline(prompt, max_length=300, do_sample=False)
    answer = response[0]['generated_text'].strip()
    return answer

def main():
    while True:
        question_kannada = input("Enter your question in Kannada (or type 'exit' to quit): ").strip()
        if question_kannada.lower() == 'exit':
            print("Exiting the QA system. Goodbye!")
            break
        if not question_kannada:
            print("Please enter a valid question.")
            continue
        
        # Translate the Kannada question to English
        try:
            question_english = translate_kannada_to_english(question_kannada)
            print(f"\n**Translated Question (English):**\n{question_english}\n")
        except Exception as e:
            print(f"Error in translation: {e}")
            continue
        
        # Generate answer in English
        try:
            answer_english = generate_answer(question_english)
            print(f"**Answer (English):**\n{answer_english}\n")
        except Exception as e:
            print(f"Error in generating answer: {e}")
            continue
        
        # Translate the English answer back to Kannada
        try:
            answer_kannada = translate_english_to_kannada(answer_english)
            print(f"**Answer (Kannada):**\n{answer_kannada}\n")
        except Exception as e:
            print(f"Error in translating answer back to Kannada: {e}")
            continue

if __name__ == "__main__":
    main()