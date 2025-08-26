import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# ----------------------------
# 1️⃣ Load COCO Captions
# ----------------------------
def load_coco_captions(data_dir="./data"):
    train_ds, val_ds, test_ds = tfds.load(
        "coco_captions",
        split=["train", "validation", "test"],
        shuffle_files=True,
        as_supervised=False,
        data_dir=data_dir
    )
    return train_ds, val_ds, test_ds


# ----------------------------
# 2️⃣ Visualize a few samples
# ----------------------------
def show_sample_images(ds, num_images=5):
    for i, example in enumerate(ds.take(num_images)):
        image = example["image"].numpy()
        caption = example["captions"][0].numpy().decode("utf-8")  # first caption
        plt.figure()
        plt.imshow(image)
        plt.title(caption)
        plt.axis("off")
        plt.show()

# ----------------------------
# 3️⃣ Precompute ResNet50 embeddings
# ----------------------------
def preprocess_and_embed(ds, batch_size=32, cache_location=None):
    resnet = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )
    resnet.trainable = False

    def preprocess(image):
        # Resize to 224x224 and preprocess for ResNet
        image = tf.image.resize(image, (224, 224))
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image

    # Map preprocessing
    ds = ds.map(lambda x: preprocess(x["image"]), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if cache_location:
        ds = ds.cache(cache_location)
    # Compute embeddings
    ds = ds.map(lambda x: resnet(x), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ----------------------------
# 4️⃣ Example usage
# ----------------------------
train_ds, val_ds, test_ds = load_coco_captions()
show_sample_images(train_ds, num_images=3)
train_embeddings = preprocess_and_embed(train_ds, batch_size=32, cache_location="./cache/train_embeddings")
