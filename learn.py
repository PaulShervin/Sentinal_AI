
from collections import defaultdict, Counter

training_text = """
python is a programming language
python is easy to learn
machine learning uses data
machine learning uses models
python uses libraries
"""

# Learn which word tends to follow each previous word.
next_word_counts = defaultdict(Counter)

for sentence in training_text.strip().splitlines():
    words = sentence.lower().split()

    for current_word, next_word in zip(words, words[1:]):
        next_word_counts[current_word][next_word] += 1

# Inspect what the model learned.
for word in ["python", "machine", "learning", "uses"]:
    print(f"\nAfter '{word}':")
    print(next_word_counts[word])

# Make a simple next-word prediction.
context = "machine learning uses".split()
last_word = context[-1]

predictions = next_word_counts[last_word]
if predictions:
    predicted_word = predictions.most_common(1)[0][0]
    print("\nContext:", " ".join(context))
    print("Predicted next word:", predicted_word)
