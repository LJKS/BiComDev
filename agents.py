import tensorflow as tf

class AgentActor(tf.keras.Model):
    def __init__(self, embed_dim=50, vocab_size=10, lstm_units=128, msg_len=3):
        super().__init__()

        self.msg_len = msg_len
        self.vocab_size = vocab_size
        self.img_embed = tf.keras.layers.Dense(embed_dim, activation='sigmoid')
        self.msg_embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.lstm_proj = tf.keras.layers.Dense(embed_dim)
        # self.msg_logits_layer = tf.keras.layers.Dense(vocab_size)
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
        cos_sim = -tf.keras.losses.cosine_similarity(img_emb, lstm_output_proj)

        # get target prediction 
        target_probs = tf.nn.sigmoid(cos_sim)  # [batch, num_images]
        
        # create output_message
        output_message = self.output_msg_dense(lstm_output)
        output_message = tf.reshape(output_message, [batch_size, self.msg_len, self.vocab_size]) # [batch_size, msg_len, vocab_size]
        output_message = tf.nn.softmax(output_message, axis=-1)

        return target_probs, output_message, (h,c)
    
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
    def __init__(self, embed_dim=50, vocab_size=10, lstm_units=128, msg_len=3):
        super().__init__()

        self.msg_len = msg_len
        self.vocab_size = vocab_size
        self.img_embed = tf.keras.layers.Dense(embed_dim, activation='sigmoid')
        self.msg_embed = tf.keras.layers.Embedding(vocab_size, embed_dim)

        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)

        self.value_head = tf.keras.layers.Dense(1)


    def call(self, feature_vector, input_message, prev_state):

        img_emb = self.img_embed(feature_vector)    # [batch_size, num_img, emb_dim]
        msg_emb = self.msg_embed(input_message)     # [batch_size, msg_len, emb_dim]

        msg_emb_reduced = tf.reduce_sum(msg_emb, axis=1)    # [batch_size, emb_dim]
        img_emb_reduced = tf.reduce_sum(img_emb, axis=1)    # [batch_size, emb_dim]

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
        print("msg_emb shape reduced seq", msg_emb_reduced.shape)
        # combine inputs to feed into lstm
        combined = tf.concat([img_emb_reduced, msg_emb_reduced], axis=2)    # [batch_size, seq_len, emb_dim*2]

        lstm_output, h, c = self.lstm(combined, initial_state=prev_state)

        val = tf.keras.layers.TimeDistributed(self.value_head)(lstm_output) 
        val = tf.squeeze(val, axis=-1) # [batch_size,seq_len]

        return val, (h,c)
    


import tensorflow as tf
import numpy as np

def test_call_on_sequence(actor_model):
    """
    Tests the call_on_sequence function of your AgentActor.
    Prints shapes and some example outputs to verify correctness.
    """
    # Define test parameters
    batch_size = 2
    seq_len = 4
    num_img = 5
    msg_len = actor_model.msg_len
    vocab_size = actor_model.vocab_size
    feat_dim = 8  # arbitrary feature vector dimension

    # Random test inputs
    feature_vector = tf.random.normal([batch_size, seq_len, num_img, feat_dim])
    input_message = tf.random.uniform([batch_size, seq_len, msg_len], minval=0, maxval=vocab_size, dtype=tf.int32)

    # Initialize LSTM state
    lstm_units = actor_model.lstm.units
    h0 = tf.zeros([batch_size, lstm_units])
    c0 = tf.zeros([batch_size, lstm_units])
    prev_state = (h0, c0)

    # Run call_on_sequence
    target_probs, output_message, (h, c) = actor_model.call_on_sequence(feature_vector, input_message, prev_state)

    # Print shapes
    print("Target probabilities shape:", target_probs.shape)  # should be [batch, seq_len, num_img]
    print("Output message shape:", output_message.shape)     # should be [batch, seq_len, msg_len, vocab_size]
    print("LSTM hidden state shape:", h.shape)               # should be [batch, lstm_units]
    print("LSTM cell state shape:", c.shape)                 # should be [batch, lstm_units]

    # Print a small slice of outputs for inspection
    print("\nExample target probabilities (first batch, first timestep):\n", target_probs[0,0,:].numpy())
    print("\nExample output message (first batch, first timestep, first token):\n", output_message[0,0,0,:].numpy())


import tensorflow as tf

def sanity_check_stepwise_vs_sequence(actor_model):
    """
    Compare call() vs call_on_sequence() outputs on the same data.
    Ensures stepwise and sequence-wise outputs are consistent.
    """
    batch_size = 2
    seq_len = 4
    num_img = 5
    msg_len = actor_model.msg_len
    vocab_size = actor_model.vocab_size
    feat_dim = 8

    # Random test data
    feature_vector = tf.random.normal([batch_size, seq_len, num_img, feat_dim])
    input_message = tf.random.uniform([batch_size, seq_len, msg_len], 0, vocab_size, dtype=tf.int32)

    # Initialize LSTM state
    lstm_units = actor_model.lstm.units
    h = tf.zeros([batch_size, lstm_units])
    c = tf.zeros([batch_size, lstm_units])
    prev_state = (h, c)

    # --- Sequence call ---
    seq_target_probs, seq_output_message, seq_state = actor_model.call_on_sequence(
        feature_vector, input_message, prev_state
    )

    # --- Stepwise call ---
    step_target_probs_list = []
    step_output_message_list = []
    state = prev_state

    for t in range(seq_len):
        fv_t = feature_vector[:, t, :, :]          # [batch, num_img, feat_dim]
        msg_t = input_message[:, t, :]             # [batch, msg_len]
        target_probs_t, output_message_t, state = actor_model.call(fv_t, msg_t, state)
        step_target_probs_list.append(target_probs_t)
        step_output_message_list.append(output_message_t)

    # Stack stepwise outputs
    step_target_probs = tf.stack(step_target_probs_list, axis=1)      # [batch, seq_len, num_img]
    step_output_message = tf.stack(step_output_message_list, axis=1)  # [batch, seq_len, msg_len, vocab_size]

    # Compare
    print("Max difference in target_probs:", tf.reduce_max(tf.abs(seq_target_probs - step_target_probs)).numpy())
    print("Max difference in output_message:", tf.reduce_max(tf.abs(seq_output_message - step_output_message)).numpy())

    # Optionally, print a slice to inspect
    print("\nSequence target_probs[0]:\n", seq_target_probs[0].numpy())
    print("\nStepwise target_probs[0]:\n", step_target_probs[0].numpy())


import tensorflow as tf

def test_critic_call_on_sequence(critic_model):
    """
    Tests the call_on_sequence function of your AgentCritic.
    Prints shapes and example outputs to verify correctness.
    """
    batch_size = 2
    seq_len = 4
    num_img = 5
    msg_len = critic_model.msg_len
    feat_dim = 8

    # Random test inputs
    feature_vector = tf.random.normal([batch_size, seq_len, num_img, feat_dim])
    input_message = tf.random.uniform([batch_size, seq_len, msg_len], 0, critic_model.vocab_size, dtype=tf.int32)

    # Initialize LSTM state
    lstm_units = critic_model.lstm.units
    h0 = tf.zeros([batch_size, lstm_units])
    c0 = tf.zeros([batch_size, lstm_units])
    prev_state = (h0, c0)

    # Run call_on_sequence
    values, (h, c) = critic_model.call_on_sequence(feature_vector, input_message, prev_state)

    # Print shapes
    print("Value outputs shape:", values.shape)       # should be [batch, seq_len]
    print("LSTM hidden state shape:", h.shape)       # [batch, lstm_units]
    print("LSTM cell state shape:", c.shape)         # [batch, lstm_units]

    # Print a small slice of outputs
    print("\nExample value outputs (first batch):", values[0].numpy())


def sanity_check_critic_stepwise_vs_sequence(critic_model):
    """
    Compares stepwise call() vs call_on_sequence() for the Critic.
    Ensures consistency of per-timestep values.
    """
    batch_size = 2
    seq_len = 4
    num_img = 5
    msg_len = critic_model.msg_len
    feat_dim = 8

    # Random test data
    feature_vector = tf.random.normal([batch_size, seq_len, num_img, feat_dim])
    input_message = tf.random.uniform([batch_size, seq_len, msg_len], 0, critic_model.vocab_size, dtype=tf.int32)

    # Initialize LSTM state
    lstm_units = critic_model.lstm.units
    h = tf.zeros([batch_size, lstm_units])
    c = tf.zeros([batch_size, lstm_units])
    prev_state = (h, c)

    # --- Sequence call ---
    seq_values, seq_state = critic_model.call_on_sequence(feature_vector, input_message, prev_state)

    # --- Stepwise call ---
    step_values_list = []
    state = prev_state

    for t in range(seq_len):
        fv_t = feature_vector[:, t, :, :]          # [batch, num_img, feat_dim]
        msg_t = input_message[:, t, :]             # [batch, msg_len]
        value_t, state = critic_model.call(fv_t, msg_t, state)
        step_values_list.append(value_t)

    step_values = tf.stack(step_values_list, axis=1)  # [batch, seq_len]

    # Compare
    print("Max difference in values:", tf.reduce_max(tf.abs(seq_values - step_values)).numpy())

    # Optionally, print example slice
    print("\nSequence values[0]:", seq_values[0].numpy())
    print("Stepwise values[0]:", step_values[0].numpy())




# Example usage:
# actor = AgentActor(embed_dim=50, vocab_size=10, lstm_units=128, msg_len=3)
# sanity_check_stepwise_vs_sequence(actor)

critic = AgentCritic(embed_dim=50, vocab_size=10, lstm_units=128, msg_len=3)

# Shape test
test_critic_call_on_sequence(critic)

# Stepwise vs sequence sanity check
sanity_check_critic_stepwise_vs_sequence(critic)
