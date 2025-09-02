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

# function for visualisation during the dev
def show_images(images):
    batch_size = images.shape[0]
    plt.figure(figsize=(batch_size * 3, 3))
    for i in range(batch_size):
        plt.subplot(1, batch_size, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()


def sample_and_embed_img(dataset, num_img=3, num_batches=5):
    show_imgs = False

    def preprocess_img(image):
        tf.debugging.assert_rank(image, 3,message="Expected image to be a 3D tensor")

        # print("dtype before resize: ", image)
        image = tf.image.resize(image, (224, 224)) # this line recasts as float32

        # so we cast back to uint8? --> Is that necessary?
        image = tf.cast(tf.round(image), tf.uint8)

        tf.debugging.assert_shapes([(image, (224, 224, 3))], message="the image needs to be of shape (224,224,3)")
        tf.debugging.assert_type(image, tf.uint8, message="Expected dtype to be uint8")
        # print("dtype after resize: ", image)

        return image
   
    embeddings_list = []
    images_list = []
    for _ in range(num_batches):
        # smaller scale buffer for shuffling cause my computer cant handle more :')
        buffer_size = 1000
        sampled_images = list(dataset.shuffle(buffer_size).take(num_img))

        images = [preprocess_img(sample["image"]) for sample in sampled_images]
        images = tf.stack([tf.convert_to_tensor(img) for img in images])

        if show_imgs == True:
            show_images(images)
        # print("images before preprocess resnet: ", images)
        images = tf.keras.applications.resnet50.preprocess_input(images)
        # print("images after preprocess resnet: ", images)
        

        resnet = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg")
        embeddings = resnet(images)  
        embeddings_list.append(embeddings)
        images_list.append(images)
   

    embeds = tf.convert_to_tensor(embeddings_list, dtype=tf.float32)
    return embeds


def create_dummy_data(data_len, num_batches):
    batch_list = []
    for _ in range(num_batches):
        indices = tf.range(0, data_len, dtype=tf.int32)
        dummy_features = tf.one_hot(indices, depth=len(indices))
        batch_list.append(dummy_features)
    dummy_data = tf.convert_to_tensor(batch_list, dtype=tf.int32)
    return dummy_data

def assign_feats_to_agents(embeddings, num_same=2, num_diff1=2, num_diff2=2):


    num_batches = tf.shape(embeddings)[0]
    num_img = tf.shape(embeddings)[1]

    # for now the splits are still hard-coded but I'll work on an function for that later
    assert num_same + num_diff1 + num_diff2 == num_img, "sum of splits must equal number of images/objects"


    indices = tf.stack([tf.random.shuffle(tf.range(num_img))[:num_img] for _ in range(num_batches)])
    batch_feats = tf.gather(embeddings, indices, batch_dims=1) # shape [num_batches, num_img, feature_dim]


    # Create mask to define which images are shared and which are different per agent
    base_mask = [0]*num_same + [1]*num_diff1 + [2]*num_diff2
    mask_list = [tf.random.shuffle(base_mask) for _ in range(num_batches)]
    mask = tf.stack(mask_list)  # [num_batches, num_img]


    # Apply mask to create agent input
    agent1_feats = tf.where(mask[..., tf.newaxis] == 2,
                            tf.zeros_like(batch_feats),  # agent1 does not see diff2
                            batch_feats)
   
    agent2_feats = tf.where(mask[..., tf.newaxis] == 1,
                            tf.zeros_like(batch_feats),  # agent2 does not see diff1
                            batch_feats)

    return agent1_feats, agent2_feats, mask


def shuffle_features_and_targets(feature_tensor, mask):
    '''extra shuffle of features to avoid position correlations (with target tracking)'''
   
    num_batches = tf.shape(feature_tensor)[0]
    num_img = tf.shape(feature_tensor)[1]

    perms = tf.argsort(tf.random.uniform((num_batches, num_img), dtype=tf.float32), axis=-1)

    batch_idx = tf.tile(tf.range(num_batches)[:, None], [1, num_img])
    gather_idx = tf.stack([batch_idx, perms], axis=-1)
    shuffled_features = tf.gather_nd(feature_tensor, gather_idx)


    new_target_positions = []
    for batch in range(num_batches):                          
        perm = tf.cast(perms[batch], tf.int32)
        orig_targets = tf.cast(tf.where(mask[batch] == 0)[:, 0], tf.int32)

        new_pos = tf.stack([tf.where(perm == t)[0][0] for t in orig_targets])
        new_target_positions.append(new_pos)

    return shuffled_features, new_target_positions, perms


# # example usage
# train_ds,_,_ = load_coco_captions(data_dir="./data")
# sample = sample_and_embed_img(train_ds, num_img=7, num_batches=5)
# #print(sample)
# a1_feats, a2_feats, mask = assign_feats_to_agents(sample, num_img=7, num_same=3, num_diff1=2, num_diff2=2)
# a1_feats_shuffled,_,_ = shuffle_features_and_targets(a1_feats, mask)

# # String dummy features for checking the shuffles
# dummy_feats = tf.constant([[
#     ["f11", "f12", "f13"],
#     ["f21", "f22", "f23"],
#     ["f31", "f32", "f33"],
#     ["f41", "f42", "f43"],
#     ["f51", "f52", "f53"],
#     ["f61", "f62", "f63"]
# ],
# [
#     ["f11", "f12", "f13"],
#     ["f21", "f22", "f23"],
#     ["f31", "f32", "f33"],
#     ["f41", "f42", "f43"],
#     ["f51", "f52", "f53"],
#     ["f61", "f62", "f63"]
# ]], dtype=tf.string)

# a1_feats, a2_feats, mask = assign_feats_to_agents(dummy_feats, num_same=2, num_diff1=2, num_diff2=2)
# a1_feats_shuffled,_,_ = shuffle_features_and_targets(a1_feats, mask)
# print(a1_feats, a1_feats_shuffled)


def get_image_input():
    train_ds, val_ds, test_ds = load_coco_captions(data_dir="./data")

    features = sample_and_embed_img(train_ds, num_img=7, num_batches=5)
    # features = create_dummy_data(data_len=7, num_batches=2)

    agent_1, agent_2, mask = assign_feats_to_agents(features, num_same=3, num_diff1=2, num_diff2=2)

    agent_1_shuffled, target_pos_1, perm_1  = shuffle_features_and_targets(agent_1, mask)
    agent_2_shuffled, target_pos_2, perm_2 = shuffle_features_and_targets(agent_2, mask)

    # I tested whether the assert would bew triggered with a misaligned target. Seems to work!
    for batch in range(tf.shape(agent_1_shuffled)[0]):
        target_feats_1 = tf.gather(agent_1_shuffled[batch], target_pos_1[batch])
        target_feats_2 = tf.gather(agent_2_shuffled[batch], target_pos_2[batch])
       
       # doesn't work for the one-hot dummy data, need to adjust later
        assert tf.reduce_all(tf.abs(target_feats_1 - target_feats_2) < 1e-6), \
            f"The targets don't align in batch {batch}!"

    # print("Agent 1 shuffled: ", agent_1_shuffled)
    return agent_1_shuffled, agent_2_shuffled

# get_image_input()
