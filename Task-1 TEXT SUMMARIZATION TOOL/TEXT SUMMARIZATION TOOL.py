# Required installations:
# pip install transformers
# pip install torch

from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization")

# Example lengthy article text
input_text = """Artificial intelligence (AI) has transformed various industries, from healthcare to finance. 
One of the most exciting areas of development is in natural language processing (NLP), 
which enables machines to understand, interpret, and generate human language. 
Recent advancements have led to powerful tools capable of performing tasks such as translation, 
sentiment analysis, and summarization. These tools are powered by deep learning models like 
Transformers, which have revolutionized how machines process language. As AI continues to evolve, 
its applications in daily life are expected to grow, enhancing productivity and enabling new innovations."""

# Generate summary
summary = summarizer(input_text, max_length=60, min_length=25, do_sample=False)

# Display original and summary
print("Original Text:\n", input_text)
print("\nSummary:\n", summary[0]['summary_text'])
