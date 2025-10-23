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


def combined_rollout_step(agent_1, critic_1, feats_a1, targets_a1, message_to_a1,
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


def merge_rollouts(all_rollouts):

    # single environment (no merge needed)
    if len(all_rollouts) == 1:
        return all_rollouts[0]

    merged = {"a1": {}, "a2": {}, "messages": {}}

    # Get keys for both agents
    a1_keys = all_rollouts[0]["a1"].keys()
    a2_keys = all_rollouts[0]["a2"].keys()

    # Merge rollout data for each agent
    for key in a1_keys:
        merged["a1"][key] = tf.concat([r["a1"][key] for r in all_rollouts], axis=0)
    for key in a2_keys:
        merged["a2"][key] = tf.concat([r["a2"][key] for r in all_rollouts], axis=0)

    if "messages" in all_rollouts[0]:
        merged["messages"] = {
            "to_a1": tf.concat([r["messages"]["to_a1"] for r in all_rollouts], axis=0),
            "to_a2": tf.concat([r["messages"]["to_a2"] for r in all_rollouts], axis=0),
        }
    return merged


def compute_gae(rewards, values, gamma=0.99, lam=0.95):

    batch_size = tf.shape(rewards)[0]
    total_timesteps = tf.shape(rewards)[1]

    advs = tf.TensorArray(tf.float32, size=total_timesteps)
    gae = tf.zeros((batch_size,), dtype=tf.float32)

    for t in tf.range(total_timesteps - 1, -1, -1):  # loop backwards
        delta = rewards[:, t] + gamma * values[:, t+1] - values[:, t]
        gae = delta + gamma * lam * gae
        advs = advs.write(t, gae)

    advs = tf.transpose(advs.stack(), [1, 0])   # (B, T)
    returns = advs + values[:, :-1]             # (B, T)

    return returns, advs


def make_train_dataset(rollout_data, buffer_size, minibatch_size):    
    def flatten(x):
        x_shape = tf.shape(x)
        return tf.reshape(x, tf.concat([[x_shape[0] * x_shape[1]], x_shape[2:]], axis=0))

    a1_flat = {k: flatten(v) for k, v in rollout_data["a1"].items()}
    a2_flat = {k: flatten(v) for k, v in rollout_data["a2"].items()}
    messages_to_a1_flat = flatten(rollout_data["messages"]["to_a1"])
    messages_to_a2_flat = flatten(rollout_data["messages"]["to_a2"])

    dataset = tf.data.Dataset.from_tensor_slices({
        "a1": a1_flat,
        "a2": a2_flat,
        "messages_to_a1": messages_to_a1_flat,
        "messages_to_a2": messages_to_a2_flat,
    })
    dataset = dataset.shuffle(buffer_size).batch(minibatch_size)
    return dataset


@tf.function
def train_step(batch, agent_1, critic_1, optimizer_agent_1, optimizer_critic_1,
               agent_2, critic_2, optimizer_agent_2, optimizer_critic_2):
    
    clip_epsilon=0.2
    entropy_coef=0.01

    with tf.GradientTape(persistent=True) as tape_a1:
        probs_img_a1, probs_msg_a1 = agent_1(batch["a1"]["features"], batch["messages_to_a1"])
        vals_a1 = critic_1(batch["a1"]["features"], batch["messages_to_a1"])

        img_dist_a1 = tfp.distributions.Categorical(probs=probs_img_a1)
        msg_dist_a1 = tfp.distributions.Bernoulli(probs=probs_msg_a1)

        img_logps_a1 = img_dist_a1.log_prob(batch["a1"]["symbols"])       
        msg_logps_a1 = msg_dist_a1.log_prob(batch["a1"]["preds"])          

        joint_logps_a1 = tf.reduce_sum(img_logps_a1, axis=-1) + tf.reduce_sum(msg_logps_a1, axis=-1)

        ratios_a1 = tf.exp(joint_logps_a1 - batch["a1"]["joint_logps"])
        entropy_a1 = tf.reduce_mean(img_dist_a1.entropy()) + tf.reduce_mean(msg_dist_a1.entropy())

        # PPO clipped surrogate loss
        unclipped_a1 = ratios_a1 * batch["a1"]["advantages"]
        clipped_a1 = tf.clip_by_value(ratios_a1, 1 - clip_epsilon, 1 + clip_epsilon)
        actor_loss_a1 = -tf.reduce_mean(tf.minimum(unclipped_a1 * batch["a1"]["advantages"], clipped_a1 * batch["a1"]["advantages"])) - entropy_coef * entropy_a1

        critic_loss_a1 = tf.reduce_mean(tf.square(batch["a1"]["returns"] - vals_a1))

    # Compute gradients
    agent1_grads = tape_a1.gradient(actor_loss_a1, agent_1.trainable_variables)
    critic1_grads = tape_a1.gradient(critic_loss_a1, critic_1.trainable_variables)

    # Apply updates
    optimizer_agent_1.apply_gradients(zip(agent1_grads, agent_1.trainable_variables))
    optimizer_critic_1.apply_gradients(zip(critic1_grads, critic_1.trainable_variables))

    del tape_a1

    with tf.GradientTape(persistent=True) as tape_a2:
        probs_img_a2, probs_msg_a2 = agent_2(batch["a2"]["features"], batch["messages_to_a2"])
        vals_a2 = critic_2(batch["a2"]["features"], batch["messages_to_a2"])

        img_dist_a2 = tfp.distributions.Categorical(probs=probs_img_a2)
        msg_dist_a2 = tfp.distributions.Bernoulli(probs=probs_msg_a2)

        img_logps_a2 = img_dist_a2.log_prob(batch["a2"]["symbols"])       
        msg_logps_a2 = msg_dist_a2.log_prob(batch["a2"]["preds"])          

        joint_logps_a2 = tf.reduce_sum(img_logps_a2, axis=-1) + tf.reduce_sum(msg_logps_a2, axis=-1)

        ratios_a2 = tf.exp(joint_logps_a2 - batch["a2"]["joint_logps"])
        entropy_a2 = tf.reduce_mean(img_dist_a2.entropy()) + tf.reduce_mean(msg_dist_a2.entropy())

        # PPO clipped surrogate loss
        unclipped_a2 = ratios_a2 * batch["a2"]["advantages"]
        clipped_a2 = tf.clip_by_value(ratios_a2, 1 - clip_epsilon, 1 + clip_epsilon)
        actor_loss_a2 = -tf.reduce_mean(tf.minimum(unclipped_a2 * batch["a2"]["advantages"], clipped_a2 * batch["a2"]["advantages"])) - entropy_coef * entropy_a2


        critic_loss_a2 = tf.reduce_mean(tf.square(batch["a2"]["returns"] - vals_a2))

    # Compute gradients
    agent2_grads = tape_a2.gradient(actor_loss_a2, agent_2.trainable_variables)
    critic2_grads = tape_a2.gradient(critic_loss_a2, critic_2.trainable_variables)

    # Apply updates
    optimizer_agent_2.apply_gradients(zip(agent2_grads, agent_2.trainable_variables))
    optimizer_critic_2.apply_gradients(zip(critic2_grads, critic_2.trainable_variables))

    del tape_a2

    return actor_loss_a1, critic_loss_a1, entropy_a1, actor_loss_a2, critic_loss_a2, entropy_a2


        


agent_1 = agents.AgentDummy()
agent_2 = agents.AgentDummy()

critic_1 = agents.AgentDummyCritic()
critic_2 = agents.AgentDummyCritic()

optimizer_agent_1 = tf.keras.optimizers.Adam(1e-2) #1e-3
optimizer_agent_2 = tf.keras.optimizers.Adam(1e-2) #1e-3
optimizer_crit_1 = tf.keras.optimizers.Adam(1e-2) #1e-3
optimizer_crit_2 = tf.keras.optimizers.Adam(1e-2) #1e-3

def train(num_iterations, num_envs, batch_size, num_minibatches, max_steps, num_epochs):
    dataset = generate_dataset(num_same=3, num_diff1=2, num_diff2=2, shuffle_buffer_size=1000, prefetch_buffer_size=1000, batch_size=batch_size, which='TRAIN')

    for iter in range(num_iterations):

        all_rollouts = []
        
        for k in range(num_envs):

            for (feats_a1, targets_a1), (feats_a2, targets_a2) in dataset.take(num_minibatches): # m rollouts per env
                
                rollout_trajectory = {
                "a1": {k: tf.TensorArray(tf.float32 if k != "symbols" else tf.int32, size=max_steps,clear_after_read=False)
                        for k in ["features", "symbols", "preds", "img_logps", "msg_logps", "joint_logps", "rewards", "values"]},
                "a2": {k: tf.TensorArray(tf.float32 if k != "symbols" else tf.int32, size=max_steps,clear_after_read=False)
                        for k in ["features", "symbols", "preds", "img_logps", "msg_logps", "joint_logps", "rewards", "values"]}
                }
                
                ta_messages_to_a1 = tf.TensorArray(tf.int32, size=max_steps)
                ta_messages_to_a2 = tf.TensorArray(tf.int32, size=max_steps)

                for current_ts in tf.range(max_steps): # collect multi-step trajectory  for each rollout
                    print("current timestep: ", current_ts)

                    message_to_a1 = tf.cond(
                        tf.equal(current_ts, 0),
                        lambda: tf.zeros([batch_size], dtype=tf.int32), # initial message just filled with zeros
                        lambda: rollout_trajectory["a2"]["symbols"].read(current_ts-1)
                        )
                    message_to_a2 = tf.cond(
                        tf.equal(current_ts, 0),
                        lambda: tf.zeros([batch_size], dtype=tf.int32), # initial message just filled with zeros
                        lambda: rollout_trajectory["a1"]["symbols"].read(current_ts-1)
                    )

                    # print("message to a1: ", message_to_a1)
                    
                    ta_messages_to_a1 = ta_messages_to_a1.write(current_ts, message_to_a1)
                    ta_messages_to_a2 = ta_messages_to_a2.write(current_ts, message_to_a2)

                    step = combined_rollout_step(agent_1, critic_1, feats_a1, targets_a1, message_to_a1,
                                                agent_2, critic_2, feats_a2, targets_a2, message_to_a2) # batch_size many rollouts?
                    
                    for key in rollout_trajectory["a1"].keys():
                        rollout_trajectory["a1"][key] = rollout_trajectory["a1"][key].write(current_ts, step["a1"][key])
                        rollout_trajectory["a2"][key] = rollout_trajectory["a2"][key].write(current_ts, step["a2"][key]) # write dict into tensor array
                    
                    # print("symbol created by a2:", rollout_trajectory["a2"]["symbols"].read(current_ts))

                    
        
                # Stack and transpose so shape is (batch_size, timesteps, ...)
                rollout_data = {
                    "a1": {k: tf.transpose(rollout_trajectory["a1"][k].stack(), [1, 0, *range(2, tf.rank(rollout_trajectory["a1"][k].stack()))])
                        for k in rollout_trajectory["a1"]},
                    "a2": {k: tf.transpose(rollout_trajectory["a2"][k].stack(), [1, 0, *range(2, tf.rank(rollout_trajectory["a2"][k].stack()))])
                        for k in rollout_trajectory["a2"]},
                    "messages": {
                        "to_a1": tf.transpose(ta_messages_to_a1.stack(), [1, 0]),
                        "to_a2": tf.transpose(ta_messages_to_a2.stack(), [1, 0]),
                    },
                }

                all_rollouts.append(rollout_data) # collected rollouts over k environments for training next
            
            print("current env: ", k)                        
        rollout_data = merge_rollouts(all_rollouts)

        last_feats_a1 = rollout_data["a1"]["features"][:, -1]
        last_feats_a2 = rollout_data["a2"]["features"][:, -1]
        last_msg_a1 = rollout_data["messages"]["to_a1"][:, -1]
        last_msg_a2 = rollout_data["messages"]["to_a2"][:, -1]

        last_val_a1 = critic_1(last_feats_a1, last_msg_a1)
        last_val_a2 = critic_2(last_feats_a2, last_msg_a2)

        all_vals_a1 = tf.concat([rollout_data["a1"]["values"], last_val_a1[:, None]], axis=1)
        all_vals_a2 = tf.concat([rollout_data["a2"]["values"], last_val_a2[:, None]], axis=1)
        
        advantages_a1, returns_a1 = compute_gae(rollout_data["a1"]["rewards"],all_vals_a1)
        advantages_a2, returns_a2 = compute_gae(rollout_data["a2"]["rewards"], all_vals_a2)

        
        rollout_data["a1"]["returns"] = returns_a1
        rollout_data["a1"]["advantages"] = advantages_a1
        rollout_data["a2"]["returns"] = returns_a2
        rollout_data["a2"]["advantages"] = advantages_a2
        print("advantages a1: ", rollout_data["a1"]["advantages"])
        
        train_dataset = make_train_dataset(rollout_data, buffer_size=1000, minibatch_size=4)

        for _ in range(num_epochs):
            for batch in train_dataset:
                actor_loss_a1, critic_loss_a1, entropy_a1, actor_loss_a2, critic_loss_a2, entropy_a2 = train_step(batch, agent_1, critic_1, optimizer_agent_1, optimizer_crit_1,
                                                                                                        agent_2, critic_2, optimizer_agent_2, optimizer_crit_2)

                tf.print("Agent 1 loss:", actor_loss_a1.numpy(), "Entropy:", entropy_a1.numpy())
                tf.print("Agent 2 loss:", actor_loss_a2.numpy(), "Entropy:", entropy_a2.numpy())
            


train(10,4,4,4,5,2)

