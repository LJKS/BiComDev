import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os

def load_coco_images(data_dir="./data"):
    train_ds, val_ds, test_ds = tfds.load(
        "coco_captions",
        split=["train", "val", "test"],
        shuffle_files=True,
        as_supervised=False,
        data_dir=data_dir
    )
    return train_ds, val_ds, test_ds


def create_dataset(dataset):
    all_feat_vs = []
    # Use dataset.map to preprocess all images
    def preprocess_img(element):
        image = element["image"]
        tf.debugging.assert_rank(image, 3,message="Expected image to be a 3D tensor")

        image = tf.image.resize(image, (224, 224)) # this line recasts as float32
        tf.debugging.assert_shapes([(image, (224, 224, 3))], message="the image needs to be of shape (224,224,3)")
        image = tf.keras.applications.resnet50.preprocess_input(image)

        return image
    
    dataset = dataset.map(preprocess_img)
    # For loop through dataset
    resnet = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg")
    for elem in dataset:
        # Use resnet to get feature vectors
        elem = tf.expand_dims(elem, axis=0) 
        feat_v = resnet(elem)  
        # Write those into list
        all_feat_vs.append(feat_v)
    # Stack list into tensor
    all_feats_tensor = tf.stack(all_feat_vs)
    all_feats_tensor = tf.squeeze(all_feats_tensor)

    # Use tf.data.dataset.from_tensor_slices to get dataset from tensor
    feat_dataset = tf.data.Dataset.from_tensor_slices(all_feats_tensor)
    # Use dataset.save to save the image vectors on hard drive
    path = os.path.join(os.getcwd(), "saved_data")

    tf.data.Dataset.save(feat_dataset, path)


    # Would it be better to switch to the resnet call you used?
    # resnet_prep_batch_size = 4
    # dataset = dataset.batch(resnet_prep_batch_size)
    # dataset = dataset.map(lambda img: resnet(img), num_parallel_calls=32)
    # dataset = dataset.unbatch()
    return feat_dataset

# sample images to use for a game
def sample_imgs(dataset, num_img):
        buffer_size = 1000
        # dataset = tf.data.Dataset.from_tensor_slices(dataset)
        sampled_images = list(dataset.shuffle(buffer_size).take(num_img))
        sampled_images = tf.convert_to_tensor(sampled_images)
        # sampled_images = tf.squeeze(sampled_images)

        return sampled_images

# assign which images are seen by which agent
def assign_feats_to_agents(embeddings, num_same=2, num_diff1=2, num_diff2=2):

    num_img = tf.shape(embeddings)[0]

    # for now the splits are still hard-coded but I'll work on an function for that later
    assert num_same + num_diff1 + num_diff2 == num_img, "sum of splits must equal number of images/objects"

    # Create mask to define which images are shared and which are different per agent
    mask = [0]*num_same + [1]*num_diff1 + [2]*num_diff2
    mask = tf.convert_to_tensor(mask)

    # mask_list = [tf.random.shuffle(base_mask) for _ in range(batch_size)]
    # mask = tf.stack(mask_list)  # [batch_size, num_img]

    # Apply mask to create agent input
    agent1_feats = tf.where(mask[..., tf.newaxis] == 2,
                            tf.zeros_like(embeddings),  # agent1 does not see diff2
                            embeddings)
    agent2_feats = tf.where(mask[..., tf.newaxis] == 1,
                            tf.zeros_like(embeddings),  # agent2 does not see diff1
                            embeddings)

    return agent1_feats, agent2_feats, mask


# shuffle the image positions per agent to avoid correlation
def shuffle_features_and_targets(feature_tensor, mask):
    '''extra shuffle of features to avoid position correlations (with target tracking)'''
   
    num_img = tf.shape(feature_tensor)[0]

    perm = tf.random.shuffle(tf.range(num_img))

    shuffled_features = tf.gather(feature_tensor, perm, axis=0)
    shuffled_mask = tf.gather(mask,perm, axis=0)

    return shuffled_features, shuffled_mask 

def get_image_input():
    train_ds, val_ds, test_ds = load_coco_images(data_dir="./data")
    data = train_ds.take(10) # creating a small subset for pipeline dev
    dataset = create_dataset(data)

    features = sample_imgs(dataset, num_img=7)
    a1_feats, a2_feats, mask = assign_feats_to_agents(features, num_same=3, num_diff1=2, num_diff2=2)

    a1_feats_shuffled, shuffled_mask_1  = shuffle_features_and_targets(a1_feats, mask)
    a2_feats_shuffled, shuffled_mask_2 = shuffle_features_and_targets(a2_feats, mask)

    # making sure targets are aligned across agents
    a1_targets = tf.boolean_mask(a1_feats_shuffled, shuffled_mask_1 == 0)
    a2_targets = tf.boolean_mask(a2_feats_shuffled, shuffled_mask_2 == 0)

    assert tf.reduce_all(tf.equal(tf.sort(a1_targets, axis=0),
                                  tf.sort(a2_targets, axis=0))), "Target are not aligned!"
    
    return a1_feats_shuffled, a2_feats_shuffled


# for pipeline testing only; not for investigating learning later
def make_dummy_embeddings(num_img=6, feat_dim=4):
    """
    Create easily-readable dummy embeddings.
    Each row = one image, each col = one feature.
    """
    return tf.reshape(tf.range(num_img * feat_dim, dtype=tf.int32),
                      (num_img, feat_dim))



a1_feats_shuffled, a2_feats_shuffled = get_image_input()

print("shuffled feats: ", a1_feats_shuffled, a2_feats_shuffled)

# TODO: - function to ranomize same/diff spilt
#       - making sure it actually works with the rollout & stuff...
#           - and figure out where and how the batching should be
#       - create correct dummy data for testing learning later



