import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Load captions
def load_captions(filename):
    with open(filename, 'r') as file:
        return file.read()

# Clean captions
import string
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

# Load data
captions = load_captions('dataset/captions.txt')
captions_dict = clean_captions(captions)

# Load features
with open('features/features.pkl', 'rb') as f:
    features = pickle.load(f)

print("Captions and features loaded!")
# Prepare all captions
all_captions = []
for key in captions_dict:
    for cap in captions_dict[key]:
        all_captions.append(cap)

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

vocab_size = len(tokenizer.word_index) + 1

print("Vocabulary Size:", vocab_size)
max_length = max(len(caption.split()) for caption in all_captions)
print("Max caption length:", max_length)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

# --------------------
# Create sequences
# --------------------
X1, X2, y = [], [], []

for img_id, captions_list in captions_dict.items():
    if img_id not in features:
        continue

    feature = features[img_id][0]

    for caption in captions_list:
        seq = tokenizer.texts_to_sequences([caption])[0]

        for i in range(1, len(seq)):
            in_seq = seq[:i]
            out_seq = seq[i]

            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)

X1 = np.array(X1)
X2 = np.array(X2)
y = np.array(y)

print("Training data prepared!")
print("X1 shape:", X1.shape)
print("X2 shape:", X2.shape)
print("y shape:", y.shape)
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model

# --------------------
# Image feature input
# --------------------
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# --------------------
# Text input
# --------------------
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

# --------------------
# Combine both
# --------------------
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)

# 🔥 IMPORTANT: NO softmax change needed
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)

# --------------------
# Compile
# --------------------
model.compile(
    loss='sparse_categorical_crossentropy',   # ✅ IMPORTANT CHANGE
    optimizer='adam'
)

print(model.summary())
# --------------------
# Train model
# --------------------
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# We need significantly more than 1 epoch to generate accurate captions instead of generic default responses.
# Adding checkpoints so training can be safely interrupted.
checkpoint = ModelCheckpoint("models/image_caption_model.h5", monitor='loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='loss', patience=3, verbose=1, restore_best_weights=True)

print("Starting deep training process... This may take a while but is REQUIRED for accurate captions!")
model.fit([X1, X2], y, epochs=2, batch_size=32, callbacks=[checkpoint, early_stopping])

print("✅ Model trained and saved!")
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved as tokenizer.pkl!")