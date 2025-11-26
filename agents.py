import tensorflow as tf

class AgentActor(tf.keras.Model):
    def __init__(self, embed_dim=50, vocab_size=10, lstm_units=128):
        super().__init__()

        self.img_embed = tf.keras.layers.Dense(embed_dim, activation='sigmoid')
        self.msg_embed = tf.keras.layers.Embedding(vocab_size, embed_dim)

        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)

        self.lstm_proj = tf.keras.layers.Dense(embed_dim)

        self.msg_logits_layer = tf.keras.layers.Dense(vocab_size)



    def call(self, feature_vector, input_message, prev_state):

        img_emb = self.img_embed(feature_vector)    # [batch_size, num_img, emb_dim]
        msg_emb = self.msg_embed(input_message)     # [batch_size, emb_dim]
       
        img_emb_reduced = tf.reduce_sum(img_emb, axis=1)    # [batch_size, emb_dim]

        # combine inputs to feed into lstm
        combined = tf.concat([img_emb_reduced, msg_emb], axis=1)    # [batch_size, emb_dim*2]
        combined = tf.expand_dims(combined, axis=1)         # [batch_size, 1, emb_dim*2]

        lstm_output, h, c = self.lstm(combined, initial_state=prev_state)
        last_output = lstm_output[:, -1, :]         # [batch, lstm_units]

        # projecting for shape match to image embedding
        lstm_output_proj = self.lstm_proj(lstm_output)

        print("lstm output shape: ", lstm_output_proj.shape)
        # compute similarity between image embedding and lstm output for target prediction
        cos_sim = -tf.keras.losses.cosine_similarity(img_emb, lstm_output_proj)

        # get target prediction 
        target_probs = tf.nn.sigmoid(cos_sim)  # [batch, num_images]
        # get logits for message
        msg_logits = self.msg_logits_layer(last_output)  # [batch, vocab_size]
        # create output_message
        output_message = tf.nn.softmax(msg_logits, axis=-1)

        return target_probs, output_message, (h,c)
    

class AgentCritic(tf.keras.Model):
    def __init__(self, embed_dim=50, vocab_size=10, lstm_units=128):
        super().__init__()

        self.img_embed = tf.keras.layers.Dense(embed_dim, activation='sigmoid')
        self.msg_embed = tf.keras.layers.Embedding(vocab_size, embed_dim)

        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)

        self.value_head = tf.keras.layers.Dense(1)


    def call(self, feature_vector, input_message, prev_state):

        img_emb = self.img_embed(feature_vector)    # [batch_size, num_img, emb_dim]
        msg_emb = self.msg_embed(input_message)     # [batch_size, emb_dim]
        # add dimension for later concat
        msg_emb = tf.expand_dims(msg_emb, axis=1)   # [batch_size, 1, emb_dim]

        # combine inputs to feed into lstm
        combined = tf.concat([img_emb, msg_emb], axis=1)

        lstm_output, h, c = self.lstm(combined, initial_state=prev_state)
        last_output = lstm_output[:, -1, :]         # [batch_size, lstm_units]

        val = tf.squeeze(self.value_head(last_output)) # [batch_size,]


        return val, (h,c)
    


