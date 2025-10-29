import tensorflow as tf
import tensorflow_datasets as tfds
import os
from tqdm import tqdm

def load_coco_images(data_dir="./data"):
    """loads mscoco image into a train, val, test split
    Args:
        data_dir (path): The locatation where the mscoco data is saved
    Return:
        A tuple containing train, val, and test datasets 
    """
    train_ds, val_ds, test_ds = tfds.load(
        "coco_captions",
        split=["train", "val", "test"], 
        shuffle_files=True,
        as_supervised=False,
        data_dir=data_dir
    )
    return train_ds, val_ds, test_ds


def save_mscoco_resnet_image_features(dataset, save_dir=""): 
    """saves the features of images from a mscoco style dataset into a local directory
    Args:
        dataset (tf.PrefetchDataset): a loaded dataset containing mscoco images
        save_dir (path): the path to the directory to be saved in
    """
    # Use dataset.map to preprocess all images
    def preprocess_img(element):
        """preprocesses mscoco style images for resnet feature extraction
        Args:
            element (dict): a single element from a mscoco style dataset 
        Return:
            a preprocessed version of the image from the element"""
        image = element["image"]
        tf.debugging.assert_rank(image, 3,message="Expected image to be a 3D tensor")
        image = tf.image.resize(image, (224, 224)) 
        tf.debugging.assert_shapes([(image, (224, 224, 3))], message="the image needs to be of shape (224,224,3)")
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image
    
    all_feat_vs = []
    dataset = dataset.map(preprocess_img)

    resnet = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg")

    # for a progressbar
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()
    # For loop through dataset
    for elem in tqdm(dataset, total=dataset_size, desc="Extracting features"):
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
    path = os.path.join(os.getcwd(), f"saved_data/{save_dir}")

    #tf.data.Dataset.save(feat_dataset, path)
    feat_dataset.save(path)

def create_local_dataset_files():
    """creates local files of the extracted image features of train, val, and test datasets"""
    train_ds, val_ds, test_ds = load_coco_images(data_dir="./data")
    save_mscoco_resnet_image_features(train_ds, save_dir="train")
    save_mscoco_resnet_image_features(test_ds, save_dir="test")
    save_mscoco_resnet_image_features(val_ds, save_dir="val")

# assign which images are seen by which agent
def assign_feats_to_agents(features, num_same=2, num_diff1=2, num_diff2=2):
    """assigns images (image features) with a split between shared and non-shared images between two agents
    Args:
        num_same (int): Number of objects that are shared between agents
        num_diff1 (int): Number of objects that are only perceived by agent 1
        num_diff2 (int): Number of objects that are only perceived by agent 2
    Returns:
        A tuple of feature tensors with num_same+num_diff1 and num_same+num_diff2 images for agent 1&2 respectively
        """

    tf.debugging.assert_equal(num_same + num_diff1 + num_diff2, tf.shape(features)[0],message="sum of splits must equal number of images/objects")

    # Create indices of which images are shared and which are different per agent
    same_idx =  tf.range(0, num_same)
    diff_1_idx = tf.range( num_same, num_same+ num_diff1)
    diff_2_idx = tf.range( num_same+ num_diff1,  num_same+num_diff1+num_diff2)

    a1_idx = tf.concat([same_idx, diff_1_idx], axis=0)
    a2_idx = tf.concat([same_idx, diff_2_idx], axis=0)

    # Apply gather according to respective indices to create agent input
    agent1_feats = tf.gather(params=features, indices=a1_idx)
    agent2_feats = tf.gather(params=features, indices=a2_idx) 

    return agent1_feats, agent2_feats


# shuffle the image positions per agent to avoid correlation
def shuffle_features_and_targets(features, num_targets):
    """extra shuffle of features to avoid position correlations (with target tracking)
    Args:
        features (tensor): image features
        num_targets (int): number of targets (shared images)
    Returns:
        A tuple of a shuffled version of the features and a target vector that details whether a feature tensor is a target  (=1) or not (=0)"""
   
    num_img = tf.shape(features)[0]
    perm = tf.random.shuffle(tf.range(num_img))

    shuffled_features = tf.gather(features, perm, axis=0)

    target_vector = perm < num_targets
    target_vector = tf.cast(target_vector, tf.float32)

    return shuffled_features, target_vector


def create_game_instances_dataset(dataset, num_same, num_diff1, num_diff2, buffer_size=1000):
    """creates a dataset that contains features, target pairs for each agent
    Args: 
        dataset (tf.PrefetchDataset): a loaded dataset containing mscoco images
        num_same (int): Number of objects that are shared between agents
        num_diff1 (int): Number of objects that are only perceived by agent 1
        num_diff2 (int): Number of objects that are only perceived by agent 2
    Return:
        A tensorflow dataset consisting of 
            ((image_features_agent1, target_vector_agent1),(image_features_agent2, target_vector_agent2)) elements
"""
    num_img = num_same + num_diff1 + num_diff2

    dataset = dataset.batch(num_img)
    dataset = dataset.map(lambda game_imgs: assign_feats_to_agents(game_imgs, num_same=num_same, num_diff1=num_diff1, num_diff2=num_diff2))
    
    # assigning (&masking) the images to each agent according to the (currently hardcoded) split
    dataset = dataset.map(lambda a1_feats, a2_feats: (shuffle_features_and_targets(a1_feats, num_same), shuffle_features_and_targets(a2_feats, num_same)))

    return dataset

