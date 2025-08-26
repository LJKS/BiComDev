import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import json
import random
# from PIL import ImageFile
from datasets import load_dataset
from functools import lru_cache

# create target from concepts (McRae et al., 2005) and get index of matching index from imagenet-1k
def create_target(labeled_concepts):
    sample1, sample2 = random.sample(list(labeled_concepts.items()), k=2)
    target, distractor = random.sample([sample1, sample2], k=2)
    target = random.sample(target[1]["sample_indices"], k=1)[0]
    distractor = random.sample(distractor[1]["sample_indices"], k=1)[0]
    target = random.randint(1,12811) #
    distractor = random.randint(1,12811) # dummy values for development, when split="train[:1%]"
    return target, distractor


# make the image from imagenet-1k useable to the VGG model
def preprocess_image(pil_img):
    img_resized = pil_img.resize((224, 224)).convert("RGB")
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.vgg16.preprocess_input(img_array)


def load_feature_extractor():
    # initializing the vgg_model outside get_feature_vector to reduce computation
    vgg_model = tf.keras.applications.VGG16(weights="imagenet", include_top=True)
    feature_extractor = tf.keras.Model(
        inputs=vgg_model.input,
        outputs=vgg_model.get_layer("fc2").output
    )

    return feature_extractor


def load_image_ds():
    images_dataset = load_dataset("NexaAIDev/ImageNet-1k", split="train[:1%]", streaming=False)
    with open("labeled_concepts.json", "r") as f:
        labeled_concepts = json.load(f)
    
    return images_dataset, labeled_concepts

# lru caching to avoid oom 
@lru_cache(maxsize=1000)
def get_feature_vector(idx, images_dataset, feature_extractor):
    try:
        pil_img = images_dataset[idx]["image"]
        img_tensor = preprocess_image(pil_img)
        features = feature_extractor(img_tensor, training=False)
        features = tf.math.l2_normalize(features, axis=-1) # normalizing features to stabilize performance
        return features # (1, 4096)
    except Exception as e:
        print(f"Error extracting features at index {idx}: {e}")
        return np.zeros((1,4096), dtype=np.float32)

