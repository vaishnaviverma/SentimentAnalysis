from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# Load model
model = load_model("sentiment_model.h5")

# Load word index
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

#FastAPI instance
app = FastAPI()

#Request body schema
class ReviewRequest(BaseModel):
    text: str

#Text preprocessing
def encode_review(text):
    words = text_to_word_sequence(text)
    encoded = [1]  # <START>
    for word in words:
        index = word_index.get(word, 2)
        if index < 10000:
            encoded.append(index)
        else:
            encoded.append(2)
    return pad_sequences([encoded], maxlen=256)

#API endpoint
@app.post("/predict")
def predict_sentiment(data: ReviewRequest):
    encoded_input = encode_review(data.text)
    prediction = model.predict(encoded_input) [0][0]
    sentiment = "Positive" if prediction>0.51 else "Negative"
    confidence = round(float(prediction),3)
    return {"Sentiment": sentiment, "Confidence": confidence}


"""

TO run: uvicorn app:app --reload

Go to http://127.0.0.1:8000/docs
for viewing the website.

"""

    