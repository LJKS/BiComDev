import tensorflow as tf
import types
CORR_TEMPERATURE = 5.0

def check_nan(x, desc):
    tf.print(desc, tf.math.reduce_any(tf.math.is_nan(x)))

class AgentActor(tf.keras.Model):
    def __init__(self, embed_dim=16, vocab_size=10, lstm_units=128, msg_len=3):
        super().__init__()
        self.msg_len = msg_len
        self.vocab_size = vocab_size
        
        self.img_embed = tf.keras.layers.Dense(embed_dim, activation='tanh')  # no activation
        self.msg_embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.lstm_proj = tf.keras.layers.Dense(embed_dim, activation='tanh') # no acitvation
        self.output_msg_dense = tf.keras.layers.Dense(vocab_size*msg_len)



    def call(self, feature_vector, input_message, prev_state):
        #check_nan(feature_vector, "feature_vector")
        #check_nan(input_message, "input_message")
        batch_size = tf.shape(feature_vector)[0]

        img_emb = self.img_embed(feature_vector)    # [batch_size, num_img, emb_dim]
        msg_emb = self.msg_embed(input_message)     # [batch_size, msg_len, emb_dim]
        #check_nan(img_emb, "img_emb")
        #check_nan(msg_emb, "msg_emb")
        msg_emb_reduced = tf.reduce_sum(msg_emb, axis=1)    # [batch_size, emb_dim]
        img_emb_reduced = tf.reduce_sum(img_emb, axis=1)    # [batch_size, emb_dim]
        #check_nan(msg_emb_reduced, "msg_emb_reduced")
        #check_nan(img_emb_reduced, "img_emb_reduced")
        # combine inputs to feed into lstm
        combined = tf.concat([img_emb_reduced, msg_emb_reduced], axis=1)    # [batch_size, emb_dim*2]
        combined = tf.expand_dims(combined, axis=1)         # [batch_size, 1, emb_dim*2]
        #check_nan(combined, "combined")
        lstm_output, h, c = self.lstm(combined, initial_state=prev_state)
        tf.print(tf.shape(prev_state), tf.shape(h), tf.shape(c))
        #check_nan(lstm_output, "lstm_output")
        #check_nan(h, "lstm_h")
        #check_nan(c, "lstm_c")

        # projecting for shape match to image embedding
        lstm_output_proj = self.lstm_proj(lstm_output) 
        #check_nan(lstm_output_proj, "lstm_output_proj")
        # compute similarity between image embedding and lstm output for target prediction
        # cos_sim = -tf.keras.losses.cosine_similarity(img_emb, lstm_output_proj, axis=2) # [batch_size, num_images]
      
        # compute correlation between image embedding and lstm output for target prediction
        #check if any elements from img_emb and lstm_output_proj are nan
        #tf.print("img_emb contains NaN:", tf.math.reduce_any(tf.math.is_nan(img_emb)))
        #tf.print("lstm_output_proj contains NaN:", tf.math.reduce_any(tf.math.is_nan(lstm_output_proj)))
        #tf.print(tf.shape(img_emb), tf.shape(lstm_output_proj))
        corr = tf.reduce_mean(img_emb * lstm_output_proj, axis=2)*CORR_TEMPERATURE
        #tf.print(corr[0,:], tf.math.sigmoid(corr[0,:]))
        # get target prediction
        # target_probs_cos = tf.nn.sigmoid(cos_sim * 5)   # [batch_size, num_images] 
        target_probs_corr = tf.nn.sigmoid(corr)
 
        # create output_message
        output_message = self.output_msg_dense(lstm_output)
        output_message = tf.reshape(output_message, [batch_size, self.msg_len, self.vocab_size]) # [batch_size, msg_len, vocab_size]
        output_message = tf.nn.softmax(output_message, axis=-1)

        return target_probs_corr, output_message, (h,c)


    def call_with_scaling(self, feature_vector, input_message, prev_state, alpha):
        #for integrated Gradients
        batch_size = tf.shape(feature_vector)[0]
        feature_vector = feature_vector * alpha
        img_emb = self.img_embed(feature_vector)    # [batch_size, num_img, emb_dim]
        msg_emb = self.msg_embed(input_message)     # [batch_size, msg_len, emb_dim]
        msg_emb = msg_emb * alpha
        msg_emb_reduced = tf.reduce_sum(msg_emb, axis=1)    # [batch_size, emb_dim]
        img_emb_reduced = tf.reduce_sum(img_emb, axis=1)    # [batch_size, emb_dim]
        #check_nan(msg_emb_reduced, "msg_emb_reduced")
        #check_nan(img_emb_reduced, "img_emb_reduced")
        # combine inputs to feed into lstm
        combined = tf.concat([img_emb_reduced, msg_emb_reduced], axis=1)    # [batch_size, emb_dim*2]
        combined = tf.expand_dims(combined, axis=1)         # [batch_size, 1, emb_dim*2]
        #check_nan(combined, "combined")
        lstm_output, h, c = self.lstm(combined, initial_state=prev_state)
        tf.print(tf.shape(prev_state), tf.shape(h), tf.shape(c))
        #check_nan(lstm_output, "lstm_output")
        #check_nan(h, "lstm_h")
        #check_nan(c, "lstm_c")

        # projecting for shape match to image embedding
        lstm_output_proj = self.lstm_proj(lstm_output)
        #check_nan(lstm_output_proj, "lstm_output_proj")
        # compute similarity between image embedding and lstm output for target prediction
        # cos_sim = -tf.keras.losses.cosine_similarity(img_emb, lstm_output_proj, axis=2) # [batch_size, num_images]

        # compute correlation between image embedding and lstm output for target prediction
        #check if any elements from img_emb and lstm_output_proj are nan
        #tf.print("img_emb contains NaN:", tf.math.reduce_any(tf.math.is_nan(img_emb)))
        #tf.print("lstm_output_proj contains NaN:", tf.math.reduce_any(tf.math.is_nan(lstm_output_proj)))
        #tf.print(tf.shape(img_emb), tf.shape(lstm_output_proj))
        corr = tf.reduce_mean(img_emb * lstm_output_proj, axis=2)*CORR_TEMPERATURE
        #tf.print(corr[0,:], tf.math.sigmoid(corr[0,:]))
        # get target prediction
        # target_probs_cos = tf.nn.sigmoid(cos_sim * 5)   # [batch_size, num_images]
        target_probs_corr = tf.nn.sigmoid(corr)

        # create output_message
        output_message = self.output_msg_dense(lstm_output)
        output_message = tf.reshape(output_message, [batch_size, self.msg_len, self.vocab_size]) # [batch_size, msg_len, vocab_size]
        output_message = tf.nn.softmax(output_message, axis=-1)

        return target_probs_corr, output_message, (h,c)

class AgentActorSeparated(tf.keras.Model):
    def __init__(self, embed_dim=16, vocab_size=10, lstm_units=128, msg_len=3):
        super().__init__()
        self.half_size = int(lstm_units/2)
        self.msg_len = msg_len
        self.vocab_size = vocab_size
        self.lstm_units = lstm_units
        self.img_embed_pred = tf.keras.layers.Dense(embed_dim, activation='tanh')
        self.msg_embed_pred = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.lstm_pred = tf.keras.layers.LSTM(self.half_size, return_sequences=True, return_state=True)
        self.lstm_proj_pred = tf.keras.layers.Dense(embed_dim, activation='tanh')

        self.img_embed_mes = tf.keras.layers.Dense(embed_dim, activation='tanh')
        self.lstm_mes = tf.keras.layers.LSTM(self.half_size, return_sequences=True, return_state=True)
        self.output_msg_dense = tf.keras.layers.Dense(vocab_size * msg_len)
        #for state initialization
        self.lstm = types.SimpleNamespace()
        self.lstm.units = lstm_units

    def call(self, feature_vector, input_message, prev_state):
        """
        Notice prev state here includes the states for both lstms! indices [:,:,0:0.5*self.lstm_units] for message, [:,:,0.5*self.lstm_units:] for prediction
        """
        def separate_state(s):
            h,c = s
            h_pred = h[:, 0:self.half_size]
            c_pred = c[:, 0:self.half_size]
            s_pred = (h_pred,c_pred)
            h_mes = h[:, self.half_size:]
            c_mes = c[:, self.half_size:]
            s_mes = (h_mes,c_mes)
            return s_pred, s_mes
        prev_state_pred, prev_state_mes = separate_state(prev_state)
        batch_size = tf.shape(feature_vector)[0]

        img_emb_pred = self.img_embed_pred(feature_vector)  # [batch_size, num_img, emb_dim]
        msg_emb_pred = self.msg_embed_pred(input_message)  # [batch_size, msg_len, emb_dim]
        msg_emb_reduced_pred = tf.reduce_sum(msg_emb_pred, axis=1)  # [batch_size, emb_dim]
        img_emb_reduced_pred = tf.reduce_sum(img_emb_pred, axis=1)  # [batch_size, emb_dim]
        combined = tf.concat([img_emb_reduced_pred, msg_emb_reduced_pred], axis=1)  # [batch_size, emb_dim*2]
        combined = tf.expand_dims(combined, axis=1)  # [batch_size, 1, emb_dim*2]

        lstm_output, h_pred, c_pred = self.lstm_pred(combined, initial_state=prev_state_pred)
        lstm_output_proj_pred = self.lstm_proj_pred(lstm_output)
        corr_pred = tf.reduce_mean(img_emb_pred * lstm_output_proj_pred, axis=2) * CORR_TEMPERATURE
        target_probs_corr = tf.nn.sigmoid(corr_pred)

        # create output_message
        img_emb_mes = self.img_embed_mes(feature_vector)  # [batch_size, num_img, emb_dim]
        img_emb_reduced_mes = tf.reduce_sum(img_emb_mes, axis=1)  # [batch_size, emb_dim]
        img_embed_t_mes = tf.expand_dims(img_emb_reduced_mes, axis=1)  # [batch_size, 1, emb_dim*2]
        lstm_output_mes, h_mes, c_mes = self.lstm_mes(img_embed_t_mes, initial_state=prev_state_mes)
        lstm_output_mes = tf.squeeze(lstm_output_mes, axis=1)
        output_message = self.output_msg_dense(lstm_output_mes)
        output_message = tf.reshape(output_message, [batch_size, self.msg_len, self.vocab_size]) # [batch_size, msg_len, vocab_size]
        output_message = tf.nn.softmax(output_message, axis=-1)
        h = tf.concat([h_pred, h_mes], axis=1)
        c = tf.concat([c_pred, c_mes], axis=1)
        return target_probs_corr, output_message, (h, c)



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
    

