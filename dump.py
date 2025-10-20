import tensorflow as tf
import tensorflow_probability as tfp
import agents
import data 
import os 

def generate_dataset(num_same, num_diff1, num_diff2, shuffle_buffer_size, prefetch_buffer_size, batch_size, which='TRAIN'):
    if which == 'TRAIN':
        path = os.path.join(os.getcwd(), "saved_data/train")
    elif which == 'TEST':
        path = os.path.join(os.getcwd(), "saved_data/test")
    elif which == 'VAL':
        path = os.path.join(os.getcwd(), "saved_data/val")

    dataset = tf.data.Dataset.load(path)
    game_input_data = data.create_game_instances_dataset(dataset, num_same, num_diff1, num_diff2, shuffle_buffer_size)

    game_input_data = game_input_data.shuffle(shuffle_buffer_size)
    game_input_data = game_input_data.batch(batch_size)
    game_input_data = game_input_data.prefetch(prefetch_buffer_size) #game_input_data = game_input_data.prefetch(tf.data.AUTOTUNE)

    return game_input_data


def combined_rollout(agent_1, critic_1, feats_a1, targets_a1, message_to_a1,
                     agent_2, critic_2, feats_a2, targets_a2, message_to_a2):
    
    # rollout agent 1
    probs_img_a1, probs_msg_a1 = agent_1(feats_a1, message_to_a1)
    vals_a1 = critic_1(feats_a1, message_to_a1)

    img_dist_a1 = tfp.distributions.Categorical(probs=probs_img_a1)
    symbols_a1 = img_dist_a1.sample() 
    img_logps_a1 =img_dist_a1.log_prob(symbols_a1)

    msg_dist_a1 = tfp.distributions.Bernoulli(probs=probs_msg_a1)
    preds_a1 = msg_dist_a1.sample()
    msg_logps_a1 = msg_dist_a1.log_prob(preds_a1)

    preds_a1 = tf.cast(preds_a1, dtype=tf.float32)

    num_targets_a1 = tf.reduce_sum(targets_a1)
    correct_a1 = tf.reduce_sum(preds_a1 * targets_a1, axis=-1)  
    rewards_a1 = correct_a1 / num_targets_a1
    joint_logps_a1 = tf.reduce_sum(img_logps_a1, axis=-1) + tf.reduce_sum(msg_logps_a1, axis=-1)

     
    # rollout agent 2
    probs_img_a2, probs_msg_a2 = agent_2(feats_a2, message_to_a2)
    vals_a2 = critic_2(feats_a2, message_to_a2)

    img_dist_a2 = tfp.distributions.Categorical(probs=probs_img_a2)
    symbols_a2 = img_dist_a2.sample() 
    img_logps_a2 = img_dist_a2.log_prob(symbols_a2)

    msg_dist_a2 = tfp.distributions.Bernoulli(probs=probs_msg_a2)
    preds_a2 = msg_dist_a2.sample()
    msg_logps_a2 = msg_dist_a2.log_prob(preds_a2)

    preds_a2 = tf.cast(preds_a2, dtype=tf.float32)

    num_targets_a2 = tf.reduce_sum(targets_a2)
    correct_a2 = tf.reduce_sum(preds_a2 * targets_a2, axis=-1)  
    rewards_a2 = correct_a2 / num_targets_a2
    joint_logps_a2 = tf.reduce_sum(img_logps_a2, axis=-1) + tf.reduce_sum(msg_logps_a2, axis=-1)

    # combine both rollouts into a single dictionary keyed to each agent
    return {
        "a1": {
            "features": feats_a1,
            "symbols": symbols_a1,
            "preds": preds_a1,
            "img_logps": img_logps_a1,
            "msg_logps": msg_logps_a1,
            "joint_logps": joint_logps_a1,
            "rewards": rewards_a1,
            "values": vals_a1,
        },
        "a2": {
            "features": feats_a2,
            "symbols": symbols_a2,
            "preds": preds_a2,
            "img_logps": img_logps_a2,
            "msg_logps": msg_logps_a2,
            "joint_logps": joint_logps_a2,
            "rewards": rewards_a2,
            "values": vals_a2,
        },
    }





agent_1 = agents.AgentDummy()
agent_2 = agents.AgentDummy()

critic_1 = agents.AgentDummyCritic()
critic_2 = agents.AgentDummyCritic()

optimizer_agent_1 = tf.keras.optimizers.Adam(1e-2) #1e-3
optimizer_agent_2 = tf.keras.optimizers.Adam(1e-2) #1e-3
optimizer_crit_1 = tf.keras.optimizers.Adam(1e-2) #1e-3
optimizer_crit_2 = tf.keras.optimizers.Adam(1e-2) #1e-3

num_iterations = 100
num_envs=4
batch_size=32
num_minibatches=32
max_steps = 10

train_dataset = generate_dataset(num_same=3, num_diff1=2, num_diff2=2, shuffle_buffer_size=1000, prefetch_buffer_size=1000, batch_size=batch_size, which='TRAIN')

for iter in range(num_iterations):
    all_rollouts = []
    for k in range(num_envs):
        rollout_trajectory = []

        for (feats_a1, targets_a1), (feats_a2, targets_a2) in train_dataset: # m rollouts per env
            for current_ts in range(max_steps): # collect multi-step trajectory  for each rollout
                message_to_a1 = tf.cond(
                    tf.equal(current_ts, 0),
                    lambda: tf.zeros([batch_size], dtype=tf.int32), # initial message just filled with zeros
                    lambda: 0# the symbols at current_ts-1 from agent_2
                    )
                message_to_a2 = tf.cond(
                    tf.equal(current_ts, 0),
                    lambda: tf.zeros([batch_size], dtype=tf.int32), # initial message just filled with zeros
                    lambda: 0# the symbols at current_ts-1 from agent_1
                )


                rollout = combined_rollout(agent_1, critic_1, feats_a1, targets_a1, message_to_a1,
                                            agent_2, critic_2, feats_a2, targets_a2, message_to_a2) # batch_size many rollouts?
                
                rollout_trajectory.append(rollout) # list of rollouts with  all values collected over the game

            all_rollouts.append(rollout_trajectory) # collected rollouts over k environments for training next
                
            rollout_data = merge_rollouts()
            advantages_a1, returns_a1 = compute_gae(rollout_data["a1"])
            advantages_a2, returns_a2 = compute_gae(rollout_data["a2"])

            
            rollout_data["a1"]["returns"] = returns_a1
            rollout_data["a1"]["advantages"] = advantages_a1
            rollout_data["a2"]["returns"] = returns_a2
            rollout_data["a2"]["advantages"] = advantages_a2






        

    # add the advantages and returns to the dictionary
