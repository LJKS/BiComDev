import tensorflow as tf
import tensorflow_datasets as tfds
import os
from tqdm import tqdm

def load_coco_images(data_dir="./data"):
    train_ds, val_ds, test_ds = tfds.load(
        "coco_captions",
        split=["train", "val", "test"], 
        shuffle_files=True,
        as_supervised=False,
        data_dir=data_dir
    )
    return train_ds, val_ds, test_ds

# Optiion rename to save_ms_coco_vectors --> no return
# als create_dataset mit cache & return
def save_mscoco_resnet_image_features(dataset, save_dir=""): # put ms_coco in name
    all_feat_vs = []
    # Use dataset.map to preprocess all images
    def preprocess_img(element):
        image = element["image"]
        tf.debugging.assert_rank(image, 3,message="Expected image to be a 3D tensor")
        image = tf.image.resize(image, (224, 224)) 
        tf.debugging.assert_shapes([(image, (224, 224, 3))], message="the image needs to be of shape (224,224,3)")
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image
    
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
    train_ds, val_ds, test_ds = load_coco_images(data_dir="./data")
    save_mscoco_resnet_image_features(train_ds, save_dir="train")
    save_mscoco_resnet_image_features(test_ds, save_dir="test")
    save_mscoco_resnet_image_features(val_ds, save_dir="val")

# assign which images are seen by which agent
def assign_feats_to_agents(embeddings, num_same=2, num_diff1=2, num_diff2=2):

    num_img = tf.shape(embeddings)[0]
    tf.debugging.assert_equal(num_same + num_diff1 + num_diff2, tf.shape(embeddings)[0],message="sum of splits must equal number of images/objects")

    # Create indices of which images are shared and which are different per agent
    same_idx =  tf.range(0, num_same)
    diff_1_idx = tf.range( num_same, num_same+ num_diff1)
    diff_2_idx = tf.range( num_same+ num_diff1,  num_same+num_diff1+num_diff2)

    a1_idx = tf.concat([same_idx, diff_1_idx], axis=0)
    a2_idx = tf.concat([same_idx, diff_2_idx], axis=0)

    # Apply gather according to respective indices to create agent input
    agent1_feats = tf.gather(params=embeddings, indices=a1_idx)
    agent2_feats = tf.gather(params=embeddings, indices=a2_idx) 

    return agent1_feats, agent2_feats


# shuffle the image positions per agent to avoid correlation
def shuffle_features_and_targets(feature_tensor, num_targets):
    '''extra shuffle of features to avoid position correlations (with target tracking)'''
   
    num_img = tf.shape(feature_tensor)[0]
    perm = tf.random.shuffle(tf.range(num_img))

    shuffled_features = tf.gather(feature_tensor, perm, axis=0)

    target_vector = perm < num_targets
    target_vector = tf.cast(target_vector, tf.float32)

    return shuffled_features, target_vector


def create_game_instances_dataset(dataset, num_same, num_diff1, num_diff2, buffer_size=1000):
    num_img = num_same + num_diff1 + num_diff2


    dataset = dataset.batch(num_img)
    dataset = dataset.map(lambda game_imgs: assign_feats_to_agents(game_imgs, num_same=num_same, num_diff1=num_diff1, num_diff2=num_diff2))
    
    # assigning (&masking) the images to each agent according to the (currently hardcoded) split
    dataset = dataset.map(lambda a1_feats, a2_feats: (shuffle_features_and_targets(a1_feats, num_same), shuffle_features_and_targets(a2_feats, num_same)))

    return dataset


def test_all():
    print("\n🔍 Running full data pipeline test...\n")

    # 1️⃣ Test dataset loading
    print("→ Loading COCO images...")
    try:
        train_ds, val_ds, test_ds = load_coco_images()
        assert isinstance(train_ds, tf.data.Dataset)
        print("✅ COCO dataset loaded successfully.")
    except Exception as e:
        print("❌ COCO load failed:", e)
        return

    # 2️⃣ Test feature extraction on a small subset
    print("\n→ Testing feature extraction on a small sample...")
    try:
        small_subset = train_ds.take(100)
        save_mscoco_resnet_image_features(small_subset, save_dir="test_sample")
        saved_path = os.path.join(os.getcwd(), "saved_data/test_sample")
        assert os.path.exists(saved_path)
        print("✅ Features extracted & saved successfully at:", saved_path)
    except Exception as e:
        print("❌ Feature extraction failed:", e)
        return

    # 3️⃣ Test reloading saved dataset
    print("\n→ Reloading saved dataset...")
    try:
        reloaded = tf.data.Dataset.load(saved_path)
        sample = next(iter(reloaded))
        print("✅ Reloaded sample shape:", sample.shape)
    except Exception as e:
        print("❌ Reloading failed:", e)
        return

    # 4️⃣ Test sampling images
    print("\n→ Testing image sampling function...")
    try:
        sampled_imgs = get_game_imgs(reloaded, num_img=7, buffer_size=10)
        assert sampled_imgs.shape[0] == 7
        print("✅ Sampling successful. Shape:", sampled_imgs.shape)
    except Exception as e:
        print("❌ Sampling failed:", e)
        return

    # 5️⃣ Test agent feature assignment
    print("\n→ Testing assign_feats_to_agents...")
    try:
        a1_feats, a2_feats = assign_feats_to_agents(sampled_imgs, num_same=3, num_diff1=2, num_diff2=2)
        print("✅ Assign successful.")
        print("   A1 feats:", a1_feats.shape)
        print("   A2 feats:", a2_feats.shape)
    except Exception as e:
        print("❌ Assign_feats_to_agents failed:", e)
        return

    # 6️⃣ Test shuffling & target creation
    print("\n→ Testing shuffle_features_and_targets...")
    try:
        shuffled_feats, targets = shuffle_features_and_targets(a1_feats, num_targets=3)
        assert shuffled_feats.shape[0] == targets.shape[0]
        num_targets_found = tf.reduce_sum(targets).numpy()
        print(f"✅ Shuffle & target creation OK. Found {num_targets_found} targets.")
    except Exception as e:
        print("❌ Shuffle_features_and_targets failed:", e)
        return

    # 7️⃣ Test game dataset creation pipeline
    print("\n→ Testing create_game_instances_dataset...")
    try:
        game_ds = create_game_instances_dataset(reloaded, 3, 2, 2, buffer_size=10)
        for a1, a2 in game_ds.take(1):
            a1_feats, a1_targets = a1
            a2_feats, a2_targets = a2
            print("✅ Game dataset instance created successfully!")
            print("   A1 feats shape:", a1_feats.shape)
            print("   A1 targets shape:", a1_targets.shape)
            print("   A2 feats shape:", a2_feats.shape)
            print("   A2 targets shape:", a2_targets.shape)
    except Exception as e:
        print("❌ create_game_instances_dataset failed:", e)
        return

    print("\n All tests passed successfully!")


