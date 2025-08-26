import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# ----------------------------
# 1️⃣ Load COCO Captions
# ----------------------------
def load_coco_captions(data_dir="./data"):
    train_ds, val_ds, test_ds = tfds.load(
        "coco_captions",
        split=["train", "val", "test"],
        shuffle_files=True,
        as_supervised=False,
        data_dir=data_dir
    )
    return train_ds, val_ds, test_ds

# ----------------------------
# 2️⃣ Visualize a few samples
# ----------------------------

# def show_sample_images(dataset, num_images=6):
#     plt.figure(figsize=(12, 6))
#     for i, example in enumerate(dataset.take(num_images)):
#         image = example["image"].numpy()
        
#         # Extract captions (list of strings)
#         captions = [c.numpy().decode("utf-8") for c in example["captions"]["text"]]

#         plt.subplot(1, num_images, i+1)
#         plt.imshow(image.astype("uint8"))
#         plt.axis("off")
#         plt.title("\n".join(captions), fontsize=6)
#     plt.show()


def sample_and_embed_img(dataset, num_img):
    buffer_size = 1000
    # 1. Take num_img samples (somwhat) random samples  ADJUST LATER
    examples = list(dataset.shuffle(buffer_size).take(num_img))

    # 2. Extract the raw images
    images = [ex["image"] for ex in examples]

    # 3. Preprocess & stack them
    images = tf.stack([tf.image.resize(img, (224, 224)) for img in images])
    images = tf.keras.applications.resnet50.preprocess_input(images)

    # 4. Get embeddings
    resnet = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg")
    embeddings = resnet(images)   # shape (3, 2048)

    return embeddings


def assign_feats_to_agents(embeddings, batch_size=1, num_img=6, num_same=2, num_diff1=2, num_diff2=2):
    assert num_same + num_diff1 + num_diff2 == num_img, "sum of splits must equal number of images/objects"

    indices = tf.stack([tf.random.shuffle(tf.range(num_img))[:num_img] for _ in range(batch_size)])
    batch_feats = tf.gather(embeddings, indices) # shape [batch_size, num_img, feature_dim]

    # Create controlled mask
    base_mask = [0]*num_same + [1]*num_diff1 + [2]*num_diff2
    mask_list = [tf.random.shuffle(base_mask) for _ in range(batch_size)]
    mask = tf.stack(mask_list)  # [batch_size, num_total]

    # Apply mask to create agent views
    agent1_feats = tf.where(mask[..., tf.newaxis] == 2,
                            tf.zeros_like(batch_feats),  # agent1 does not see diff2
                            batch_feats)
    
    agent2_feats = tf.where(mask[..., tf.newaxis] == 1,
                            tf.zeros_like(batch_feats),  # agent2 does not see diff1
                            batch_feats)

    return agent1_feats, agent2_feats, mask



