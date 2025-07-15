import nltk
import os

# Download the Punkt tokenizer models
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize

# Define your paragraph
paragraph = "This is the first sentence. This is the second sentence. This is the third sentence."

# Tokenize the paragraph into sentences
sentences = sent_tokenize(paragraph)

# Print the sentences
for sentence in sentences:
    print(sentence)
