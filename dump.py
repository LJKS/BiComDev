import tensorflow as tf


class AgentActor(tf.keras.Model):
    def __init__(self, embed_dim=50, vocab_size=10):
        super().__init__()
        # Dense layer for image input
        self.embed_img = tf.keras.layers.Dense(embed_dim, activation='sigmoid')  
        self.embed_msg = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)

        # LSTM to have access to previous messages
        self.lstm_proj = tf.keras.layers.Dense(2048)
        self.lstm = tf.keras.layers.LSTM(embed_dim, return_state=True)
        
        self.similarity_head = tf.keras.layers.Dense(1, activation=None)


        self.msg_logits = tf.keras.layers.Dense(vocab_size)   # Next message distribution
        self.target_head = tf.keras.layers.Dense(1, activation="sigmoid")  # Target prediction



    def call(self, feature_vector, input_message, prev_state=None):
        
        img_emb = self.embed_img(feature_vector)
        msg_emb = self.embed_msg(input_message)
        msg_emb = tf.expand_dims(msg_emb, axis=1)  # shape: (batch, 1, embed_dim)


        lstm_output, h, c = self.lstm(msg_emb, initial_state=prev_state)
        lstm_output = self.lstm_proj(lstm_output)
        similarity = -tf.keras.losses.cosine_similarity(img_emb, lstm_output, axis=-1)

        # input_message = tf.expand_dims(input_message, axis=1)
        # input_message = tf.expand_dims(input_message, axis=-1)

        # if prev_state != None:
        #    prev_state
            
        combined = tf.concat([img_emb, lstm_output, similarity], axis=-1)

        msg_logits = self.msg_logits(combined)
        msg_probs = tf.nn.softmax(msg_logits)

        target_pred = self.target_head(combined)
        
        return msg_probs, target_pred, (h, c)


class AgentCritic(tf.keras.Model):
    def __init__(self, embed_dim=50, vocab_size=10):
        super().__init__()
        # Dense layer for image input
        self.embed_img = tf.keras.layers.Dense(embed_dim, activation='sigmoid')  
        self.embed_msg = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)

        # LSTM to have access to previous messages
        self.lstm = tf.keras.layers.LSTM(embed_dim, return_state=True)
        
        self.value_head = tf.keras.layers.Dense(1)

        self.msg_logits = tf.keras.layers.Dense(vocab_size)   # Next message distribution
        self.target_head = tf.keras.layers.Dense(1, activation="sigmoid")  # Target prediction



    def call(self, feature_vector, input_message, prev_state=None):
        
        img_emb = self.embed_img(feature_vector)
        msg_emb = self.embed_msg(input_message)
        msg_emb = tf.expand_dims(msg_emb, axis=1)  # shape: (batch, 1, embed_dim)

        lstm_output, h, c = self.lstm(msg_emb, initial_state=prev_state)

        similarity = -tf.keras.losses.cosine_similarity(img_emb, lstm_output, axis=-1)

        combined = tf.concat([img_emb, lstm_output, similarity], axis=-1)

        value = self.value_head(combined)  
        value = tf.squeeze(value, axis=-1) 

        return value, (h, c)
        



