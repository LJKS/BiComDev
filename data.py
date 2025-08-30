import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt




def load_coco_captions(data_dir="./data"):
    train_ds, val_ds, test_ds = tfds.load(
        "coco_captions",
        split=["train", "val", "test"],
        shuffle_files=True,
        as_supervised=False,
        data_dir=data_dir
    )
    return train_ds, val_ds, test_ds


def sample_and_embed_img(dataset, num_img=3, num_batches=5):


    def preprocess_img(image):
        tf.debugging.assert_rank(
            image, 3,
            message="Expected image to be a 3D tensor (H, W, C)"
        )


        # print("dtype before resize: ", image)
        image = tf.image.resize(image, (224, 224)) # this line recasts as float32


        image = tf.cast(tf.round(image), tf.uint8)


        tf.debugging.assert_shapes([(image, (224, 224, 3))], message="the image needs to be of shape (224,224,3)")
        tf.debugging.assert_type(image, tf.uint8, message="Expected dtype to be uint8")
        # print("dtype after resize: ", image)
       
        return image
   
    embeddings_list = []
    for batch in range(num_batches):
        # smaller scale buffer for shuffling cause my computer cant handle more :')
        buffer_size = 100
        examples = list(dataset.shuffle(buffer_size).take(num_img))


        images = [preprocess_img(ex["image"]) for ex in examples]
        images = tf.stack([tf.convert_to_tensor(img) for img in images])


        print("images before preprocess resnet: ", images)
        images = tf.keras.applications.resnet50.preprocess_input(images)
        print("images after preprocess resnet: ", images)


        resnet = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg")
        embeddings = resnet(images)  
        embeddings_list.append(embeddings)
   
    embeds = tf.convert_to_tensor(embeddings_list, dtype=tf.float32)
    return embeds


train_ds,_,_ = load_coco_captions(data_dir="./data")
sample = sample_and_embed_img(train_ds, num_img=3, num_batches=5)
print(sample)


def show_images_from_batch(images, captions=None):
    batch_size = images.shape[0]
    plt.figure(figsize=(batch_size * 3, 3))
    for i in range(batch_size):
        plt.subplot(1, batch_size, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")
        # if captions is not None:
        #     plt.title(captions[i], fontsize=8)
    plt.show()


def create_dummy_data(data_len):
    indices = tf.range(0, data_len, dtype=tf.int32)
    dummy_data = tf.one_hot(indices, depth=len(indices))
    return dummy_data


def assign_feats_to_agents(embeddings, batch_size=1, num_img=6, num_same=2, num_diff1=2, num_diff2=2):
    assert num_same + num_diff1 + num_diff2 == num_img, "sum of splits must equal number of images/objects"


    indices = tf.stack([tf.random.shuffle(tf.range(num_img))[:num_img] for _ in range(batch_size)])
    batch_feats = tf.gather(embeddings, indices) # shape [batch_size, num_img, feature_dim]


    # Create controlled mask
    base_mask = [0]*num_same + [1]*num_diff1 + [2]*num_diff2
    mask_list = [tf.random.shuffle(base_mask) for _ in range(batch_size)]
    mask = tf.stack(mask_list)  # [batch_size, num_total]


    # Apply mask to create agent input
    agent1_feats = tf.where(mask[..., tf.newaxis] == 2,
                            tf.zeros_like(batch_feats),  # agent1 does not see diff2
                            batch_feats)
   
    agent2_feats = tf.where(mask[..., tf.newaxis] == 1,
                            tf.zeros_like(batch_feats),  # agent2 does not see diff1
                            batch_feats)
   


    return agent1_feats, agent2_feats, mask




def shuffle_features_and_targets(feature_tensor, mask):
   
    batch_size = tf.shape(feature_tensor)[0]
    num_img    = tf.shape(feature_tensor)[1]


    perms = tf.argsort(tf.random.uniform((batch_size, num_img), dtype=tf.float32), axis=-1)  # (B, num_img)


    batch_idx = tf.tile(tf.range(batch_size)[:, None], [1, num_img])       # (B, num_img)
    gather_idx = tf.stack([batch_idx, perms], axis=-1)                     # (B, num_img, 2)
    shuffled_features = tf.gather_nd(feature_tensor, gather_idx)           # (B, num_img, feat_dim)


    new_target_positions = []
    for b in range(batch_size):                          
        perm_b = tf.cast(perms[b], tf.int32)
        orig_targets = tf.cast(tf.where(mask[b] == 0)[:, 0], tf.int32)


        new_pos = tf.stack([tf.where(perm_b == t)[0][0] for t in orig_targets])
        new_target_positions.append(new_pos)


    return shuffled_features, new_target_positions, perms






def finish_input():


    train_ds, val_ds, test_ds = load_coco_captions(data_dir="./data")


    features = sample_and_embed_img(train_ds, num_img=3, batch_size=5, buffer_size=100, show_images=True)
    # features = create_dummy_data(data_len=10)
    agent_1, agent_2, mask = assign_feats_to_agents(features, batch_size=5, num_img=3, num_same=1, num_diff1=1, num_diff2=1)


    agent_1_shuffled, target_pos_1, perm_1  = shuffle_features_and_targets(agent_1, mask)
    agent_2_shuffled, target_pos_2, perm_2 = shuffle_features_and_targets(agent_2, mask)


    # I tested whether the assert would bew triggered with a misaligned target. Seems to work!
    for b in range(tf.shape(agent_1_shuffled)[0]):
        target_feats_1 = tf.gather(agent_1_shuffled[b], target_pos_1[b])
        target_feats_2 = tf.gather(agent_2_shuffled[b], target_pos_2[b])
       
        assert tf.reduce_all(tf.abs(target_feats_1 - target_feats_2) < 1e-6), \
            f"The targets don't align in batch {b}!"


    print("Agent 1 shuffled: ", agent_1_shuffled)
    return agent_1_shuffled, agent_2_shuffled






# finish_input()
