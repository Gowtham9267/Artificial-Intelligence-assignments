pip install transformers torch nltk
from transformers import pipeline
import nltk
nltk.download('punkt')

# Load a pre-trained summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_article(article_text, max_length=150, min_length=30):
    # Split into manageable chunks if the text is very long
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(article_text)
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
