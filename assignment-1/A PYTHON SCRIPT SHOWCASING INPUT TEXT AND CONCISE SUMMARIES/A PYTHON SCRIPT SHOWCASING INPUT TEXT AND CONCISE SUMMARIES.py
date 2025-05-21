pip install transformers torch nltk
# text_summarizer.py

from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK tokenizer models (only needed the first time)
nltk.download('punkt')

# Load summarization pipeline using BART model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_article(text, max_length=130, min_length=30):
    sentences = sent_tokenize(text)
    current_chunk = ""
    summary = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 1000:
            current_chunk += " " + sentence
        else:
            summary_piece = summarizer(current_chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            summary += summary_piece + " "
            current_chunk = sentence

    if current_chunk:
        summary_piece = summarizer(current_chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        summary += summary_piece

    return summary.strip()

if __name__ == "__main__":
    print("\n--- TEXT SUMMARIZER TOOL ---")
    print("Paste or type your article below. Press Enter twice to submit.\n")

    # Collect multi-line input
    input_lines = []
    while True:
        line = input()
        if line == "":
            break
        input_lines.append(line)

    full_text = " ".join(input_lines)
    
    print("\nGenerating summary...\n")
    summary = summarize_article(full_text)
    print("--- SUMMARY ---")
    print(summary)
