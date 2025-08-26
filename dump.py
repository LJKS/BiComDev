# # import tensorflow as tf
# # import numpy as np
# # import random as rnd
# # import tensorflow_datasets as tfds

import tensorflow as tf

def generate_agent_features(all_features, batch_size=4, num_total=6,
                            num_same=2, num_diff1=2, num_diff2=2):

    assert num_same + num_diff1 + num_diff2 == num_total, "Sum of splits must equal total number of images/objects"

    # Sample features for each batch
    indices = tf.stack([tf.random.shuffle(tf.range(num_total))[:num_total] for _ in range(batch_size)])
    batch_features = tf.gather(all_features, indices)  # shape [B, num_total, F]

    # Create controlled mask
    base_mask = [0]*num_same + [1]*num_diff1 + [2]*num_diff2
    mask_list = [tf.random.shuffle(base_mask) for _ in range(batch_size)]
    mask = tf.stack(mask_list)  # [batch_size, num_total]

    # Apply mask to create agent views
    agent1_feats = tf.where(mask[..., tf.newaxis] == 2,
                            tf.zeros_like(batch_features),  # agent1 does not see diff2
                            batch_features)
    
    agent2_feats = tf.where(mask[..., tf.newaxis] == 1,
                            tf.zeros_like(batch_features),  # agent2 does not see diff1
                            batch_features)

    return agent1_feats, agent2_feats, mask

# --------------------------
# Example usage
# --------------------------
# String-based dummy features
dummy_feats = tf.constant([
    ["f11", "f12", "f13"],
    ["f21", "f22", "f23"],
    ["f31", "f32", "f33"],
    ["f41", "f42", "f43"],
    ["f51", "f52", "f53"],
    ["f61", "f62", "f63"]
], dtype=tf.string)

agent1, agent2, mask = generate_agent_features(dummy_feats, batch_size=1, num_total=6,
                                              num_same=2, num_diff1=2, num_diff2=2)

print("Agent1 feats:", agent1)
print("Agent2 shape:", agent2.shape)
print("Mask shape:", mask.shape)
print("Mask example:\n", mask)

# import tensorflow as tf

# def generate_agent_features(all_features, batch_size=4, num_total=6):
#     """
#     Generate agent feature sets with random assignments of same, diff1, diff2.
    
#     Args:
#         all_features: tf.Tensor of shape [N, feature_dim]
#         batch_size: number of sets per batch
#         num_total: total features per set (e.g., 6)

#     Returns:
#         agent1_feats: [batch_size, num_total, feature_dim]
#         agent2_feats: [batch_size, num_total, feature_dim]
#         mask: [batch_size, num_total] 0=same, 1=diff1, 2=diff2
#     """
#     N, feature_dim = all_features.shape

#     # Sample features for each batch
#     # indices = tf.stack([tf.random.shuffle(tf.range(N))[:num_total] for _ in range(batch_size)])
#     # batch_features = tf.gather(all_features, indices)  # [B, num_total, F]

#     # hard coded features for better clarity in debugging

#     dummy_feats = tf.constant([
#         ["f11", "f12", "f13"],
#         ["f21", "f22", "f23"],
#         ["f31", "f32", "f33"],
#         ["f41", "f42", "f43"],
#         ["f51", "f52", "f53"],
#         ["f61", "f62", "f63"]
#     ], dtype=tf.string)

#     # Random mask: 0=same, 1=diff1, 2=diff2
#     # mask = tf.random.uniform((batch_size, num_total), minval=0, maxval=3, dtype=tf.int32)

#     base = [0]*num_same + [1]*num_diff1 + [2]*num_diff2
#     masks = []
#     for _ in range(batch_size):
#         masks.append(tf.random.shuffle(base))
#     mask = tf.stgack(masks)

#     # Create agent1 and agent2 views
#     # "same" features are shared; "diff1" goes to agent1; "diff2" goes to agent2
#     agent1_feats = tf.where(mask[..., tf.newaxis] == 2,
#                             tf.zeros_like(dummy_feats),  # agent1 does not see diff2
#                             dummy_feats)
    
#     agent2_feats = tf.where(mask[..., tf.newaxis] == 1,
#                             tf.zeros_like(dummy_feats),  # agent2 does not see diff1
#                             dummy_feats)

#     return agent1_feats, agent2_feats, mask

# # --------------------------
# # Example usage
# # --------------------------
# all_features = tf.random.normal((1000, 4096))
# agent1, agent2, mask = generate_agent_features(all_features, batch_size=4, num_total=6)

# print("Agent1 shape:", agent1)  # (4, 6, 4096)
# print("Agent2 shape:", agent2.shape)  # (4, 6, 4096)
# print("Mask shape:", mask.shape)      # (4, 6)


# def get_raw_train_test_val_ds():
#     """
#     Load the raw dataset split into train, test, and validation sets.

#     Returns:
#         tuple: A tuple containing the raw training, testing, and validation datasets.
#     """

#     (train_data, test_data, val_data) = tfds.load("coco_captions",
#                                                          data_dir="./data",
#                                                          split=["train", "test", "val"]
#                                                          )
#     return train_data, test_data, val_data



# def load_and_preprocess_data(ds, cache_location=None, resnet_prep_batch_size=4):
#     resnet = tf.keras.applications.ResNet50(weights="imagenet",
#                                        include_top=False,
#                                        pooling="avg")
#     resnet.trainable = False

#     def preprocess_data(element):
#         #expected as single elements, not batched
#         tf.debugging.assert_rank(element["image"], 3,
#                                  message="Expected image needs to be a 3D tensor (height, width, channels)")

#         # Resize and preprocess image.
#         image = element["image"]
#         # if project_settings.DATASET_DEBUG:
#         #     tf.print(f"Image shape after extraction: {image.shape}")
#         # image = tf.image.resize(image, (224, 224))
#         # if project_settings.DATASET_DEBUG:
#         #     tf.print(f"Image shape after resize: {image.shape}")
#         image = tf.keras.applications.resnet50.preprocess_input(image)
#         return image

#     # Test preprocess_data() with a sample.
#     sample_element = next(iter(ds))
#     preprocessed_image = preprocess_data(sample_element)

#     tf.debugging.assert_equal(preprocessed_image.shape, (224, 224, 3), message=f"Expected output shape (224, 224, 3), got {preprocessed_image.shape}")

#     # Preprocess the datasets.
#     ds = ds.map(preprocess_data, num_parallel_calls=2)
#     ds = ds.batch(resnet_prep_batch_size)
#     ds = ds.map(lambda img: resnet(img), num_parallel_calls=32)
#     ds = ds.unbatch()
#     if not cache_location == None:
#         ds = ds.cache(cache_location)
#     return ds

# train_ds, test_ds, val_ds = get_raw_train_test_val_ds()

# processed = load_and_preprocess_data(train_ds)

# print("This is what the loaded dataset looks like:", processed)
# # def shuffle_tensors(tensors_dict): # or list or stack?
# #     # slots = ["same1", "same2", "diff1_1", "diff1_2", "diff2_1", "diff2_2"]
   
# #     keys = list(tensors_dict.keys())
# #     rnd.shuffle(keys)

# #     # assignment = {slot: tensors_dict[k] for slot, k in zip(slots, keys)}
# #     assignment = {
# #         "same":   [(key, tensors_dict[key]) for key in keys[0:2]],
# #         "diff1":  [(key, tensors_dict[key]) for key in keys[2:4]],
# #         "diff2":  [(key, tensors_dict[key]) for key in keys[4:6]],
# #     }

# #     return assignment


# # # create the needed shape (B, image_features_1, image_features_2, image_features_3, image_features_4, message)
# # def create_dummy_data(num_of_img=6):
# #     img_features = []
# #     for img in range(num_of_img):
# #         img_features.append(np.maximum(0, np.random.normal(loc=0.0, scale=1.0, size=(4096,))))

# #     img_f1 = img_features[0]
# #     img_f2 = img_features[1]
# #     img_f3 = img_features[2]
# #     img_f4 = img_features[3]
# #     img_f5 = img_features[4]
# #     img_f6 = img_features[5]

# #     img_f1 = tf.convert_to_tensor(img_f1, dtype=tf.float32)
# #     img_f2 = tf.convert_to_tensor(img_f2, dtype=tf.float32)
# #     img_f3 = tf.convert_to_tensor(img_f3, dtype=tf.float32)
# #     img_f4 = tf.convert_to_tensor(img_f4, dtype=tf.float32)
# #     img_f5 = tf.convert_to_tensor(img_f5, dtype=tf.float32)
# #     img_f6 = tf.convert_to_tensor(img_f6, dtype=tf.float32)

# #     # all_img_f = tf.concat([img_f1, img_f2,img_f3, img_f4], axis=-1)
# #     # all_img_f = tf.convert_to_tensor(img_features, dtype=tf.float32)

# #     # Store in a dictionary
# #     tensors = {
# #         "img_f1": img_f1,
# #         "img_f2": img_f2,
# #         "img_f3": img_f3,
# #         "img_f4": img_f4,
# #         "img_f5": img_f5,
# #         "img_f6": img_f6

# #     }

# #     shuffled = shuffle_tensors(tensors)

# #     # Extract tensors from each pair
# #     same_tensors  = [tensor for name, tensor in shuffled["same"]]
# #     diff1_tensors = [tensor for name, tensor in shuffled["diff1"]]
# #     diff2_tensors = [tensor for name, tensor in shuffled["diff2"]]

# #     # Optional: unpack them for convenience
# #     same1, same2     = same_tensors
# #     diff1_1, diff1_2 = diff1_tensors
# #     diff2_1, diff2_2 = diff2_tensors


# #     # same_img_pair = "something"
# #     # diff_img_pair_1 = "something not A"
# #     # diff_img_pair_2 = "something not B"


# # # How can I make sure that the same/diff pairs are created properly?

# # # swap = tf.random.uniform((batch_size,)) < 0.5
# # #         target_feats = tf.squeeze(target_feats) # shape (B, 4096)
# # #         distractor_feats = tf.squeeze(distractor_feats) # shape (B, 4096)
# # #         input_left = tf.where(swap[:, tf.newaxis], target_feats, distractor_feats)
# # #         input_right = tf.where(swap[:, tf.newaxis], distractor_feats, target_feats)
# # #         correct_choice = tf.where(swap, tf.zeros((batch_size,), dtype=tf.int64),
# # #                                     tf.ones((batch_size,), dtype=tf.int64))
# #     return shuffled
  
# # img = create_dummy_data(6)
# # for slot, items in img.items():
# #     print(f"\n{slot} pair:")
# #     for name, tensor in items:
# #         print(f"  came from {name}, value={tensor.numpy()}")
# # class Agent(tf.keras.Model):
# #     def __init__(self, embed_dim=50, vocab_size=10):
# #         super().__init__()
# #         # self.embed = tf.keras.layers.Dense(embed_dim, activation='sigmoid') # should be sigmoid to mirror paper
# #         # self.vocab_logits = tf.keras.layers.Dense(vocab_size)

# #     def call(self, img_f1,img_f2, img_f3, img_f4, message):

       
# #         return 
    