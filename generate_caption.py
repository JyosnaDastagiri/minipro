import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
import pickle

# Load model
from tensorflow.keras.models import load_model


model = load_model("models/image_caption_model.h5")



# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load max length
max_length = 38  # use same as training

# Load feature extractor
base_model = InceptionV3(weights='imagenet')
model_new = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def extract_feature(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feature = model_new.predict(img, verbose=0)
    return feature

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption(model, tokenizer, photo, max_length):
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    return in_text

# Test image
image_path = "test.jpg"   # put your image here
photo = extract_feature(image_path)

caption = generate_caption(model, tokenizer, photo, max_length)
caption = caption.replace("startseq", "").replace("endseq", "").strip()
print("Generated Caption:", caption)