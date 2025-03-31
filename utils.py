# utils.py
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# Load IMDb word index
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

def encode_review(text, maxlen=256, vocab_size=10000):
    words = text_to_word_sequence(text)
    encoded = [1]  # <START>
    for word in words:
        index = word_index.get(word, 2)
        if index < vocab_size:
            encoded.append(index)
        else:
            encoded.append(2)
    return pad_sequences([encoded], maxlen=maxlen)
