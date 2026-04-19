# ============================================================
# N-GRAM LANGUAGE MODEL
# ============================================================
# Author      : Vanam Naveen Kumar
# Dataset     : IMDB Movie Reviews (25,000 reviews via Keras)
# Description : Statistical language model built from scratch
#               using N-gram probabilities to predict the next
#               word in a sentence and evaluate model performance
#               through perplexity metrics.
#
# Run this in Google Colab cell by cell as marked below.
# ============================================================


# ============================================================
# CELL 1: IMPORTING LIBRARIES
# ============================================================
# re           : regular expressions for text cleaning
# random       : selecting random test sentences
# math         : log and exp for perplexity calculation
# pickle       : saving and loading the trained model
# tqdm         : progress bar while building n-gram model
# imdb         : built-in keras dataset (25k movie reviews)
# ============================================================

import re
import random
import math
import pickle
from tqdm import tqdm
from tensorflow.keras.datasets import imdb


# ============================================================
# CELL 2: LOADING IMDB DATASET
# ============================================================
# Keras IMDB dataset contains 25,000 movie reviews for training
# and 25,000 for testing, all pre-tokenized as integers.
#
# Steps:
#   1. Download word index (integer → word mapping)
#   2. Load encoded reviews (each word is an integer)
#   3. Decode integers back to readable text
#   4. Split each review into sentences by comma
#   5. Filter out very short fragments (< 3 words)
# ============================================================

# Word index maps each word to a unique integer
word_index = imdb.get_word_index()

# Reverse it: integer → word (for decoding)
reverse_word_index = {v: k for k, v in word_index.items()}

# Load encoded reviews (only training data needed for building model)
(x_train, _), (x_test, _) = imdb.load_data()

def decode_review(encoded):
    # Indices are offset by 3 (0=padding, 1=start, 2=unknown)
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded])

print("Decoding training reviews...")
raw_reviews = [decode_review(review) for review in x_train]
print(f"Total reviews loaded: {len(raw_reviews)}")

# Split reviews into sentences
data = []
for review in raw_reviews:
    sentences = review.split(",")
    for sent in sentences:
        sent = sent.strip()
        if len(sent.split()) >= 3:
            data.append(sent)

print(f"Total sentences extracted: {len(data)}")
print(f"Sample sentence: {data[0]}")


# ============================================================
# CELL 3: TEXT CLEANING
# ============================================================
# Raw text contains noise like HTML tags, punctuation,
# extra spaces, and uppercase letters.
#
# Steps:
#   1. Remove HTML tags like <br />
#   2. Remove punctuation characters
#   3. Collapse multiple spaces into one
#   4. Convert everything to lowercase
#   5. Strip leading/trailing whitespace
#   6. Remove any sentences that became empty after cleaning
# ============================================================

reg = r"<br\s*/>|\'s|\.\.+|[!@#$%^&*()_\-=+{}\[\]:;'\"<>,/`~]"

def tagremover(sent):
    sent = re.sub(reg, " ", sent)           # remove special characters
    sent = re.sub(r'\s+', ' ', sent)        # collapse multiple spaces
    return sent.lower().strip()             # lowercase and trim

data = [tagremover(s) for s in data]
data = [s for s in data if s != ""]        # drop empty strings

print(f"Cleaned sentences: {len(data)}")
print(f"Sample: {data[0]}")


# ============================================================
# CELL 4: N-GRAM MODEL CLASS
# ============================================================
# An N-gram model predicts the next word based on the previous
# (n-1) words, called the "history" or "context".
#
# Example for trigram (n=3):
#   Sentence  : "the movie was great"
#   Becomes   : "<s> <s> the movie was great </s>"
#   Trigrams  : (<s>,<s>)→the, (<s>,the)→movie,
#               (the,movie)→was, (movie,was)→great
#
# Two dictionaries are built:
#   gram_count : counts how often each (history, word) pair appears
#   hist_count : counts how often each history appears
#
# These two counts are later used to calculate probabilities:
#   P(word | history) = count(history, word) / count(history)
# ============================================================

class N_gram:
    def __init__(self, data, n):
        self.data = data
        self.n = n

    def gramDiv(self):
        gram_count = dict()     # stores (history, word) → frequency
        hist_count = dict()     # stores history → frequency

        for sent in tqdm(self.data, desc=f"Building {self.n}-gram model"):
            if self.n == 1:
                # Unigram: no history, just count each word
                sentence = ("<s> " + sent + " </s>").split()
                for word in sentence:
                    gram_count[(word,)] = gram_count.get((word,), 0) + 1
                hist_count[('',)] = hist_count.get(('',), 0) + len(sentence)
            else:
                # N-gram: sliding window of size n over the sentence
                sentence = ("<s> " * (self.n - 1) + sent + " </s>").split()
                history = sentence[:self.n - 1]     # first (n-1) words as starting history
                for j in range(self.n - 1, len(sentence)):
                    current_word = sentence[j]
                    key = (tuple(history), current_word)
                    gram_count[key] = gram_count.get(key, 0) + 1
                    hist_count[tuple(history)] = hist_count.get(tuple(history), 0) + 1
                    history.pop(0)                  # slide window: remove oldest word
                    history.append(current_word)    # add current word to history

        return gram_count, hist_count


# ============================================================
# CELL 5: BUILD, VERIFY AND SAVE MODEL
# ============================================================
# This cell does 4 things:
#   1. Builds the n-gram frequency counts using the class above
#   2. Verifies the key structure is correct for chosen n
#   3. Calculates raw probabilities for all n-grams
#   4. Saves the model to disk as a .pkl file
#
# Probability formula (Maximum Likelihood Estimation):
#   P(word | history) = count(history, word) / count(history)
#
# Note: This is raw MLE probability. Smoothing is applied
# later during perplexity calculation, not stored here.
# ============================================================

n = int(input("Enter n (1=unigram, 2=bigram, 3=trigram): "))
print(f"\nBuilding {n}-gram model...")
print(f"History size: {n-1} word(s) per context")

# Build model
obj = N_gram(data, n)
gram_n1_count, gram_n0_count = obj.gramDiv()

# Verify key structure
print("\nVerification — first 3 keys in gram_n1_count:")
for key in list(gram_n1_count.keys())[:3]:
    print(f"  {key}")

print("\nFirst 3 keys in gram_n0_count:")
for key in list(gram_n0_count.keys())[:3]:
    print(f"  {key}")

if n > 1:
    sample_key = list(gram_n1_count.keys())[0]
    history_len = len(sample_key[0])
    print(f"\nHistory length in keys : {history_len}")
    print(f"Expected history length: {n-1}")
    if history_len == n - 1:
        print("✅ Model structure CORRECT — proceeding")
    else:
        print("❌ Model structure WRONG — restart runtime and re-run from Cell 1")
        raise ValueError(f"Expected history length {n-1}, got {history_len}.")
else:
    print("✅ Unigram model — structure check skipped")

# Build vocabulary from all unique words in corpus
all_words = set()
for sent in data:
    all_words.update(sent.split())
all_words.update(["<s>", "</s>"])
vocabulary_size = len(all_words)
print(f"\nVocabulary size: {vocabulary_size}")

# Calculate MLE probabilities
def calculate_prob(gram_n0_count, gram_n1_count, n):
    ngram_probs = dict()
    for key, count in gram_n1_count.items():
        if n == 1:
            history = ('',)
        else:
            history = key[0]
        denom = gram_n0_count.get(history, 0)
        if denom > 0:
            ngram_probs[key] = count / denom
    return ngram_probs

ngram_probs = calculate_prob(gram_n0_count, gram_n1_count, n)
print(f"Total n-gram probabilities: {len(ngram_probs)}")

# Save model
with open(f"ngram_model_n{n}.pkl", "wb") as f:
    pickle.dump((gram_n1_count, gram_n0_count, ngram_probs, vocabulary_size, n), f)
print(f"\n✅ Model saved as ngram_model_n{n}.pkl")

print("\n" + "="*50)
print(f"Model Summary")
print("="*50)
print(f"  n value         : {n}")
print(f"  Vocabulary size : {vocabulary_size}")
print(f"  Unique n-grams  : {len(gram_n1_count)}")
print(f"  Probabilities   : {len(ngram_probs)}")
print("="*50)


# ============================================================
# CELL 6: LOAD SAVED MODEL (USE ON RE-RUNS)
# ============================================================
# Instead of rebuilding the model from scratch every time,
# we can load the saved .pkl file directly.
#
# IMPORTANT: Only use this cell if Cell 5 was already run
# in the current Colab session and the .pkl file exists.
# If runtime was restarted, skip this and re-run Cell 5.
# ============================================================

# Uncomment below to load instead of rebuild:

# n = int(input("Enter n value of saved model to load: "))
# with open(f"ngram_model_n{n}.pkl", "rb") as f:
#     gram_n1_count, gram_n0_count, ngram_probs, vocabulary_size, n = pickle.load(f)
# print("="*50)
# print(f"Model loaded successfully!")
# print(f"  n value         : {n}")
# print(f"  Vocabulary size : {vocabulary_size}")
# print(f"  Unique n-grams  : {len(gram_n1_count)}")
# print(f"  Probabilities   : {len(ngram_probs)}")
# print("="*50)


# ============================================================
# CELL 7: PERPLEXITY EVALUATION
# ============================================================
# Perplexity measures how "surprised" the model is by a sentence.
# Lower perplexity = model understands the sentence better.
#
# Formula:
#   Perplexity = exp( -1/N * Σ log P(wᵢ | history) )
#
# Where N = number of words in sentence
#
# Laplace (Add-1) Smoothing is applied here to handle unseen
# n-grams. Without smoothing, any unseen trigram would give
# probability 0, making the entire sentence probability 0.
#
# Smoothed probability:
#   P(word | history) = (count(history, word) + 1)
#                       / (count(history) + vocabulary_size)
# ============================================================

def get_ngrams_for_sentence(sentence, n):
    sentence = tagremover(sentence)
    tokens = ("<s> " * (n - 1) + sentence + " </s>").split()
    ngrams = []

    if n == 1:
        for word in tokens:
            ngrams.append((('',), word))
    else:
        history = tokens[:n - 1]
        for j in range(n - 1, len(tokens)):
            current = tokens[j]
            ngrams.append((tuple(history), current))
            history.pop(0)
            history.append(current)

    return ngrams, tokens


def calculate_perplexity(sentence, n, gram_n1_count, gram_n0_count, vocabulary_size):
    ngrams, tokens = get_ngrams_for_sentence(sentence, n)
    log_prob_sum = 0

    for (history, word) in ngrams:
        if n == 1:
            numerator   = gram_n1_count.get((word,), 0) + 1
            denominator = gram_n0_count.get(('',), 0) + vocabulary_size
        else:
            numerator   = gram_n1_count.get((history, word), 0) + 1
            denominator = gram_n0_count.get(history, 0) + vocabulary_size

        prob = numerator / denominator
        log_prob_sum += math.log(prob)      # use log to avoid underflow

    # Perplexity formula
    perplexity    = math.exp(-log_prob_sum / len(ngrams))
    sentence_prob = math.exp(log_prob_sum)
    return perplexity, sentence_prob


# Load test reviews
(_, _), (x_test, _) = imdb.load_data()

test_data = []
for review in x_test:
    decoded = decode_review(review)
    for sent in decoded.split(","):
        sent = tagremover(sent.strip())
        if len(sent.split()) >= 3:
            test_data.append(sent)

print(f"Test sentences loaded: {len(test_data)}")

# Evaluate on random test sentence
test_sentence = random.choice(test_data)
print(f"\nTest sentence: {test_sentence}")

perplexity, sent_prob = calculate_perplexity(
    test_sentence, n, gram_n1_count, gram_n0_count, vocabulary_size
)

print("\n" + "="*50)
print(f"Sentence probability : {sent_prob:.2e}")
print(f"Perplexity           : {perplexity:.4f}")
print("="*50)

# Sanity check — common sentences should score lower perplexity
test_cases = [
    "the movie was great",
    "i really enjoyed this film",
    "the acting was terrible",
    "this is a good movie",
    "the story was very boring"
]

print(f"\n{'Sentence':<45} {'Perplexity':>12}")
print("-" * 60)
for sent in test_cases:
    pp, sp = calculate_perplexity(sent, n, gram_n1_count, gram_n0_count, vocabulary_size)
    print(f"{sent:<45} {pp:>12.2f}")


# ============================================================
# CELL 8: INTERACTIVE NEXT WORD PREDICTOR
# ============================================================
# Given a text input, predicts the top-5 most likely next words
# using the trained n-gram probability table.
#
# How it works:
#   1. Clean and tokenize the input text
#   2. Extract the last (n-1) words as history
#   3. Look up all n-grams that match this history
#   4. Sort by probability and return top-k predictions
#
# For unigram (n=1): ignores input, returns most frequent words
# For bigram  (n=2): uses last 1 word as context
# For trigram (n=3): uses last 2 words as context
# ============================================================

def predict_next_word(input_text, n, ngram_probs, top_k=5):
    input_text = tagremover(input_text)
    tokens = input_text.split()

    if n == 1:
        # Unigram has no context — just return most frequent words
        candidates = {
            key[0]: prob
            for key, prob in ngram_probs.items()
            if key[0] not in ["<s>", "</s>"]
        }
    else:
        if len(tokens) < n - 1:
            print(f"Please enter at least {n-1} word(s) for a {n}-gram model.")
            return []
        history = tuple(tokens[-(n - 1):])      # take last (n-1) words as context
        candidates = {
            key[1]: prob
            for key, prob in ngram_probs.items()
            if key[0] == history and key[1] not in ["<s>", "</s>"]
        }

    if not candidates:
        print("No predictions found. Try different words.")
        return []

    # Sort by probability descending and return top k
    return sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]


# Interactive prediction loop
print("="*50)
print(f"Next Word Predictor — {n}-gram model")
print(f"Tip: Enter at least {n-1} word(s) as context")
print("Type 'quit' to exit")
print("="*50)

while True:
    user_input = input("\nEnter your text: ").strip()
    if user_input.lower() == 'quit':
        print("Exiting.")
        break
    if not user_input:
        continue

    predictions = predict_next_word(user_input, n, ngram_probs, top_k=5)

    if predictions:
        print(f"\nTop predictions after '{user_input}':")
        print("-" * 40)
        for rank, (word, prob) in enumerate(predictions, 1):
            bar = "█" * max(1, int(prob * 50))
            print(f"  {rank}. {word:<15} {prob:.6f}  {bar}")
