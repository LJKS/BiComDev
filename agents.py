import tensorflow as tf

class AgentActor(tf.keras.Model):
    def __init__(self, embed_dim=16, vocab_size=10, lstm_units=128, msg_len=3):
        super().__init__()
        self.msg_len = msg_len
        self.vocab_size = vocab_size
        
        self.img_embed = tf.keras.layers.Dense(embed_dim)  # no activation
        self.msg_embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.lstm_proj = tf.keras.layers.Dense(embed_dim) # no acitvation
        self.output_msg_dense = tf.keras.layers.Dense(vocab_size*msg_len)



    def call(self, feature_vector, input_message, prev_state):

        batch_size = tf.shape(feature_vector)[0]

        img_emb = self.img_embed(feature_vector)    # [batch_size, num_img, emb_dim]
        msg_emb = self.msg_embed(input_message)     # [batch_size, msg_len, emb_dim]

        msg_emb_reduced = tf.reduce_sum(msg_emb, axis=1)    # [batch_size, emb_dim]
        img_emb_reduced = tf.reduce_sum(img_emb, axis=1)    # [batch_size, emb_dim]

        # combine inputs to feed into lstm
        combined = tf.concat([img_emb_reduced, msg_emb_reduced], axis=1)    # [batch_size, emb_dim*2]
        combined = tf.expand_dims(combined, axis=1)         # [batch_size, 1, emb_dim*2]

        lstm_output, h, c = self.lstm(combined, initial_state=prev_state) 

        # projecting for shape match to image embedding
        lstm_output_proj = self.lstm_proj(lstm_output) 

        # compute similarity between image embedding and lstm output for target prediction
        # cos_sim = -tf.keras.losses.cosine_similarity(img_emb, lstm_output_proj, axis=2) # [batch_size, num_images]
      
        # compute correlation between image embedding and lstm output for target prediction
        corr = tf.reduce_sum(img_emb * lstm_output_proj, axis=2)

        # get target prediction 
        # target_probs_cos = tf.nn.sigmoid(cos_sim * 5)   # [batch_size, num_images] 
        target_probs_corr = tf.nn.sigmoid(corr)
 
        # create output_message
        output_message = self.output_msg_dense(lstm_output)
        output_message = tf.reshape(output_message, [batch_size, self.msg_len, self.vocab_size]) # [batch_size, msg_len, vocab_size]
        output_message = tf.nn.softmax(output_message, axis=-1)

        return target_probs_corr, output_message, (h,c)
    
    def call_on_sequence(self, feature_vector, input_message, prev_state):
        batch_size = tf.shape(feature_vector)[0]
        seq_len = tf.shape(feature_vector)[1]

        img_emb = tf.keras.layers.TimeDistributed(self.img_embed)(feature_vector) # [batch_size, seq_len, num_img, emb_dim]
        msg_emb = tf.keras.layers.TimeDistributed(self.msg_embed)(input_message)  # [batch_size, seq_len, msg_len, emb_dim]

        msg_emb_reduced = tf.reduce_sum(msg_emb, axis=2)    # [batch_size, emb_dim]
        img_emb_reduced = tf.reduce_sum(img_emb, axis=2)    # [batch_size, emb_dim]

        combined = tf.concat([img_emb_reduced, msg_emb_reduced], axis=2)    # [batch_size, seq_len, emb_dim*2]
        
        lstm_output, h, c = self.lstm(combined, initial_state=prev_state)

        # projecting for shape match to image embedding
        lstm_output_proj = tf.keras.layers.TimeDistributed(self.lstm_proj)(lstm_output) # [batch_size, seq_len, emb_dim]
        
        lstm_output_proj = tf.expand_dims(lstm_output_proj, axis=2)  # [batch_size, seq_len, 1, emb_dim]
        img_norm = tf.nn.l2_normalize(img_emb, axis=-1)
        lstm_norm = tf.nn.l2_normalize(lstm_output_proj, axis=-1)

        # compute similarity between image embedding and lstm output for target prediction
        cos_sim = tf.reduce_sum(img_norm * lstm_norm, axis=-1)  # [batch_size, seq_len, num_img]


        # get target prediction 
        target_probs = tf.nn.sigmoid(cos_sim)  # [batch_size, seq_len, num_images]
        
        # create output_message
        output_message = tf.keras.layers.TimeDistributed(self.output_msg_dense)(lstm_output)
        output_message = tf.reshape(output_message, [batch_size, seq_len, self.msg_len, self.vocab_size]) # [batch_size, seq_len, msg_len, vocab_size]
        output_message = tf.nn.softmax(output_message, axis=-1)

        return target_probs, output_message, (h,c)




class AgentCritic(tf.keras.Model):
    def __init__(self, embed_dim=16, vocab_size=10, lstm_units=128):
        super().__init__()

        self.vocab_size = vocab_size
        self.img_embed = tf.keras.layers.Dense(embed_dim)
        self.msg_embed = tf.keras.layers.Embedding(vocab_size, embed_dim)

        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)

        self.value_head = tf.keras.layers.Dense(1)


    def call(self, feature_vector, input_message, prev_state):

        img_emb = self.img_embed(feature_vector)    # [batch_size, num_img, emb_dim]
        msg_emb = self.msg_embed(input_message)     # [batch_size, msg_len, emb_dim]

        msg_emb_reduced = tf.reduce_sum(msg_emb, axis=1)        # [batch_size, emb_dim]
        img_emb_reduced = tf.reduce_sum(img_emb, axis=1)        # [batch_size, emb_dim]

        # combine inputs to feed into lstm
        combined = tf.concat([img_emb_reduced, msg_emb_reduced], axis=1)    # [batch_size, emb_dim*2]
        combined = tf.expand_dims(combined, axis=1)         # [batch_size, 1, emb_dim*2]

        lstm_output, h, c = self.lstm(combined, initial_state=prev_state)

        val = tf.squeeze(self.value_head(lstm_output)) # [batch_size,]

        return val, (h,c)
    
    def call_on_sequence(self, feature_vector, input_message, prev_state):
        
        img_emb = self.img_embed(feature_vector)    # [batch_size, seq_len, num_img, emb_dim]
        msg_emb = self.msg_embed(input_message)     # [batch_size, seq_len, msg_len, emb_dim]

        msg_emb_reduced = tf.reduce_sum(msg_emb, axis=2)    # [batch_size, seq_len, emb_dim]
        img_emb_reduced = tf.reduce_sum(img_emb, axis=2)    # [batch_size, seq_len, emb_dim]
        # combine inputs to feed into lstm
        combined = tf.concat([img_emb_reduced, msg_emb_reduced], axis=2)    # [batch_size, seq_len, emb_dim*2]

        lstm_output, h, c = self.lstm(combined, initial_state=prev_state)

        val = tf.keras.layers.TimeDistributed(self.value_head)(lstm_output) 
        val = tf.squeeze(val, axis=-1) # [batch_size,seq_len]

        return val, (h,c)
    

