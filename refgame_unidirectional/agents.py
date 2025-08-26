# import numpy as np
import tensorflow as tf
# import tensorflow_probability as tfp
# import json
# import random
# from PIL import ImageFile
# from datasets import load_dataset
# from functools import lru_cache
# import pickle



class AgnosticSender(tf.keras.Model):
    def __init__(self, embed_dim=50, vocab_size=10, temperature=10.0):
        super().__init__()
        self.temperature = temperature
        self.embed = tf.keras.layers.Dense(embed_dim, activation='sigmoid') # should be sigmoid to mirror paper
        self.vocab_logits = tf.keras.layers.Dense(vocab_size)

    def call(self, input_target, input_distractor):
        # print("Sender shape of input: ", input_target.shape, input_distractor.shape)

        x = tf.concat([input_target, input_distractor], axis=-1)
        x = self.embed(x)
        logits = self.vocab_logits(x)
        return tf.nn.softmax(logits / self.temperature) # Gibbs distribution
    
class SenderCritic(tf.keras.Model):
    def __init__(self, embed_dim=50):
        super().__init__()
        self.embed = tf.keras.layers.Dense(embed_dim, activation='sigmoid')
        self.value_head = tf.keras.layers.Dense(1)

    def call(self, input_target, input_distractor):
        x = tf.concat([input_target, input_distractor], axis=-1)
        h = self.embed(x)
        return tf.squeeze(self.value_head(h), axis=-1)  # [B]



class Receiver(tf.keras.Model):
    def __init__(self, embed_dim=50, vocab_size=10, temperature=10.0):
        super().__init__()
        self.temperature = temperature
        self.embed_img = tf.keras.layers.Dense(embed_dim, activation='sigmoid') # should be sigmoid to mirror paper
        self.embed_symbol = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)

    def call(self, input_left, input_right, input_message):
        # print("Receiver shape of input: ", input_left.shape, input_right.shape, input_message.shape)
        left_emb = self.embed_img(input_left)
        right_emb = self.embed_img(input_right)
        message_emb = self.embed_symbol(input_message)#[:, 0, :]
        dot_left = tf.reduce_sum(message_emb * left_emb, axis=-1, keepdims=True)
        dot_right = tf.reduce_sum(message_emb * right_emb, axis=-1, keepdims=True)
        logits = tf.concat([dot_left, dot_right], axis=-1)
        return tf.nn.softmax(logits / self.temperature) # Gibbs distribution


class ReceiverCritic(tf.keras.Model):
    def __init__(self, embed_dim=50, vocab_size=10):
        super().__init__()
        self.embed_img = tf.keras.layers.Dense(embed_dim, activation='sigmoid')
        self.embed_symbol = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.value_head = tf.keras.layers.Dense(1)

    def call(self, input_left, input_right, input_message):
        left_emb = self.embed_img(input_left)
        right_emb = self.embed_img(input_right)
        message_emb = self.embed_symbol(input_message)#[:, 0, :]

        # Use full embeddings, not dot products
        h = tf.concat([left_emb, right_emb, message_emb], axis=-1)
        value = tf.squeeze(self.value_head(h), axis=-1)
        return value



