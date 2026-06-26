import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

print("Loading...")
model = load_model("models/image_caption_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

base_model = InceptionV3(weights='imagenet')
model_new = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
max_length = 38

def extract_feature(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return model_new.predict(img, verbose=0)

def generate_caption(photo):
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = None
        for w, idx in tokenizer.word_index.items():
            if idx == yhat:
                word = w
                break
        if word is None: break
        in_text += " " + word
        if word == "endseq": break
    return in_text.replace("startseq ", "").replace(" endseq", "")

# print("Testing dummy.jpg:", generate_caption(extract_feature("c:/Users/D.keerthi Reddy/.gemini/antigravity/scratch/image_captioner/backend/dummy.jpg")))
print("Testing test.jpg:", generate_caption(extract_feature("test.jpg")))
