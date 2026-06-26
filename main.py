# def load_captions(filename):
#     file = open(filename, 'r')
#     text = file.read()
#     file.close()
#     return text

# captions = load_captions('dataset/captions.txt')
# print(captions[:500])   # print first 500 characters
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

captions = load_captions('dataset/captions.txt')
captions_dict = clean_captions(captions)

# print sample
for key, value in list(captions_dict.items())[:1]:
    print(key)
    print(value)