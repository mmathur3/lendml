from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("summarization")

# Sample text to summarize
text = """
Transformers are a type of model architecture in machine learning that have gained significant attention 
for their effectiveness in natural language processing tasks. They rely on a mechanism called self-attention 
to weigh the importance of different words in a sentence, allowing for better understanding and generation 
of human-like text. Since their introduction in 2017 with the paper "Attention is All You Need," transformers 
have been used in various applications, from chatbots to translation and text summarization.
"""

# Summarize the text
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)

# Print the summary
print("Summary:", summary[0]['summary_text'])
