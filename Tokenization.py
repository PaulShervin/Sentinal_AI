
from collections import Counter

text = """
I love Python
I love machine learning
Python uses libraries
I love coding
"""

# Step 1: Convert text into lowercase words
tokens = text.lower().split()

# Step 2: Find unique words and assign IDs
unique_words = sorted(set(tokens))

vocabulary = {
    word: token_id
    for token_id, word in enumerate(unique_words)
}

# Step 3: Convert the text into token IDs
token_ids = [vocabulary[word] for word in tokens]

print("Tokens:", tokens)
print("\nVocabulary:", vocabulary)
print("\nToken IDs:", token_ids)
 