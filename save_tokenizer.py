import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
import string

def load_captions(filename):
    with open(filename, 'r') as file:
        return file.read()

def clean_captions(text):
    captions_dict = {}
    for line in text.split('\n'):
        if len(line) < 2:
            continue
        parts = line.split(',')
        image_id = parts[0].split('#')[0]
        caption = " ".join(parts[1:])
        caption = caption.lower()
        caption = caption.translate(str.maketrans('', '', string.punctuation))
        if image_id not in captions_dict:
            captions_dict[image_id] = []
        captions_dict[image_id].append("startseq " + caption + " endseq")
    return captions_dict

print("Loading captions...")
captions = load_captions('dataset/captions.txt')
captions_dict = clean_captions(captions)

all_captions = []
for key in captions_dict:
    for cap in captions_dict[key]:
        all_captions.append(cap)

print("Fitting tokenizer...")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

print("Saving tokenizer...")
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Tokenizer saved!")
