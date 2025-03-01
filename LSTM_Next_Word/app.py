import pickle
import streamlit as st

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open("hamlet.txt") as f:
    data = f.read().lower()

def predict_next_word(model, tokenizer, seed_text, maxseq):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    if len(token_list) >= maxseq:
        token_list = token_list[-maxseq-1:]
    token_list = pad_sequences([token_list], maxlen=maxseq-1, padding="pre")
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word 
    return None

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

st.title('Next Word Prediction')
st.write('This is a simple next word prediction app using LSTM model.')

# input_text = "This is an red fruit called "
# print(f"Input text: {input_text}")

# print(f"Next word: {next_words}")


input_text = st.text_input('Enter a sentence:', 'I am feeling')

max_sq_len = model.input_shape[1]+1
next_words = predict_next_word(model, tokenizer, input_text, max_sq_len)

if st.button("Predict Next Word!", type="primary"):
    st.write("Next Word:", next_words)
