import tensorflow
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
model = load_model('rnn_model.h5')


def decode_review(review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(reveiw):
    preprocessed_review = preprocess_text(reveiw)
    prediction = model.predict(preprocessed_review)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative" 
    return sentiment, prediction[0][0]


example = "The movie was good. The animation and the graphics were out of the world. The story was amazing. I loved it."
sentiment, confidence = predict_sentiment(example)
sentiment, confidence