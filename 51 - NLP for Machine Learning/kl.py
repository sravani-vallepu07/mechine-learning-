import nltk
from nltk.stem import WordNetLemmatizer
import os

# Find the NLTK data directory
nltk_data_dir = nltk.data.path[0]

# Delete the WordNet corpus
wordnet_path = os.path.join(nltk_data_dir, 'corpora', 'wordnet.zip')
if os.path.exists(wordnet_path):
    os.remove(wordnet_path)

# Re-download the WordNet corpus
nltk.download('wordnet')

# Create a WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Test the lemmatizer
print(lemmatizer.lemmatize("going"))
