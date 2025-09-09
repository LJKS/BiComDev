import tensorflow as tf

# intstantiation of the agent networks, actor and critic PPO framework
# - both agents should have the same structure for symmetry (just as two humans in a conversation have (broadly) the same level of cognitive functions at their disposal)


# agent_1 actor

# agent_1 critic

# agent_2 actor

# agent_2 critic


# adjusted agents from unidirectional version as dummy agents

class AgentDummy(tf.keras.Model):
    def __init__(self, embed_dim=50, vocab_size=10, temperature=10.0):
        super().__init__()
        self.temperature = temperature
        self.embed_img = tf.keras.layers.Dense(embed_dim, activation='sigmoid') # should be sigmoid to mirror paper
        self.vocab_logits = tf.keras.layers.Dense(vocab_size)
        self.embed_symbol = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)


    def call(self, feature_vector, input_message=None):
        
        # def sender_fn():
        #     img_emb = self.embed_img(feature_vector)
        #     pooled = tf.reduce_mean(img_emb, axis=1)  # attention pooling could go here later
        #     logits = self.vocab_logits(pooled)
        #     return tf.nn.softmax(logits / self.temperature)

        # def receiver_fn():
        #     img_emb = self.embed_img(feature_vector)  
        #     msg_emb = self.embed_symbol(input_message) 
        #     msg_emb = tf.expand_dims(msg_emb, axis=1)   

        #     dot = tf.reduce_sum(img_emb * msg_emb, axis=-1)  
        #     return tf.nn.sigmoid(dot) # tf.nn.softmax(dot / self.temperature)
        
        def combined():
            img_emb = self.embed_img(feature_vector)
            pooled = tf.reduce_mean(img_emb, axis=1)  # attention pooling could go here later
            logits = self.vocab_logits(pooled)
            logits_sm = tf.nn.softmax(logits / self.temperature)

            msg_emb = self.embed_symbol(input_message) 
            msg_emb = tf.expand_dims(msg_emb, axis=1)   

            dot = tf.reduce_sum(img_emb * msg_emb, axis=-1)  
            dot_sig = tf.nn.sigmoid(dot) # tf.nn.softmax(dot / self.temperature)

            return logits_sm, dot_sig

        return combined()
        # return tf.cond(tf.equal(role, 0), sender_fn, receiver_fn)

class AgentDummyCritic(tf.keras.Model):
    def __init__(self, embed_dim=50, vocab_size=10):
        super().__init__()
        self.embed_img = tf.keras.layers.Dense(embed_dim, activation="sigmoid")
        self.embed_symbol = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.value_head = tf.keras.layers.Dense(1)

    def call(self, feature_tensor, input_message):
        img_emb = self.embed_img(feature_tensor)
        pooled = tf.reduce_mean(img_emb, axis=1)

        msg_emb = self.embed_symbol(input_message)
        h = tf.concat([pooled, msg_emb], axis=-1)

        value = tf.squeeze(self.value_head(h), axis=-1)  # (B,)
        return value

    
        # def sender_crit_fn():
        #     img_emb = self.embed_img(feature_tensor)
        #     pooled = tf.reduce_mean(img_emb, axis=1) 
        #     value = tf.squeeze(self.value_head(pooled), axis=-1)
        #     return value

        # def receiver_crit_fn():
        #     img_emb = self.embed_img(feature_tensor)       
        #     pooled = tf.reduce_mean(img_emb, axis=1)       
        #     msg_emb = self.embed_symbol(input_message)     
        #     h = tf.concat([pooled, msg_emb], axis=-1)     
        #     value = tf.squeeze(self.value_head_receiver(h), axis=-1)
        #     return value

        # return tf.cond(tf.equal(role, 0), sender_crit_fn, receiver_crit_fn)


