import spacy

# Load the English NLP model from spaCy
nlp = spacy.load("en_core_web_sm")

# Sample text to extract nouns from
text = """
Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think 
and learn like humans. AI systems can perform tasks that typically require human intelligence, such as understanding 
natural language, recognizing patterns, and making decisions. The field of AI encompasses various subfields, including 
machine learning, natural language processing, and robotics. As technology continues to advance, AI is being integrated 
into a wide range of applications, from healthcare to finance, enhancing efficiency and driving innovation.
"""

# Process the text
doc = nlp(text)

# Extract nouns
nouns = [token.text for token in doc if token.pos_ == "NOUN"]

# Print the list of nouns
print("Nouns:", nouns)
