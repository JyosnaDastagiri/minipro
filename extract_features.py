from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import numpy as np
import os
from tqdm import tqdm
import pickle

# Load model
model = InceptionV3(weights='imagenet')
model = Model(inputs=model.input, outputs=model.layers[-2].output)

def extract_features(directory):
    features = {}

    for img_name in tqdm(os.listdir(directory)):
        img_path = os.path.join(directory, img_name)

        image = load_img(img_path, target_size=(299, 299))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        feature = model.predict(image, verbose=0)
        features[img_name] = feature

    return features

features = extract_features('dataset/images')

# save features
with open('features/features.pkl', 'wb') as f:
    pickle.dump(features, f)

print("Features extracted successfully!")
