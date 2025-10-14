import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import agents 
import data 


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


    
def rollout(agent, critic, features, targets, message):

    probs_img, probs_msg = agent(features, message)
    vals = critic(features, message)

    img_dist = tfp.distributions.Categorical(probs=probs_img)
    symbols = img_dist.sample() # (B,)
    img_logps = img_dist.log_prob(symbols) # (B,)

    msg_dist = tfp.distributions.Bernoulli(probs=probs_msg)
    preds = msg_dist.sample()
    msg_logps = msg_dist.log_prob(preds)

    preds = tf.cast(preds, dtype=tf.float32)
    # normalized by target_num to avoid reward inflation from higher number of targets

    num_targets = tf.reduce_sum(targets)

    correct = tf.reduce_sum(preds * targets, axis=-1)  
    rewards = correct / num_targets

    joint_logps = tf.reduce_sum(img_logps, axis=-1) + tf.reduce_sum(msg_logps, axis=-1)

    return symbols, preds, img_logps, msg_logps, joint_logps, rewards, vals



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


@tf.function
def train_step(agent, critic, optimizer_agent, optimizer_critic,
               features, messages,
               symbols, preds, 
               old_joint_logps, 
               advantages, returns):
    
    clip_epsilon=0.2
    entropy_coef=0.01

    with tf.GradientTape(persistent=True) as tape:
        probs_img, probs_msg = agent(features, messages)
        vals = critic(features, messages)

        img_dist = tfp.distributions.Categorical(probs=probs_img)
        msg_dist = tfp.distributions.Bernoulli(probs=probs_msg)

        img_logps = img_dist.log_prob(symbols)       
        msg_logps = msg_dist.log_prob(preds)          

        joint_logps = tf.reduce_sum(img_logps, axis=-1) + tf.reduce_sum(msg_logps, axis=-1)

        ratios = tf.exp(joint_logps - old_joint_logps)
        entropy = tf.reduce_mean(img_dist.entropy()) + tf.reduce_mean(msg_dist.entropy())

        # PPO clipped surrogate loss
        unclipped = ratios * advantages
        clipped = tf.clip_by_value(ratios, 1 - clip_epsilon, 1 + clip_epsilon)
        actor_loss = -tf.reduce_mean(tf.minimum(unclipped * advantages, clipped * advantages)) - entropy_coef * entropy

        critic_loss = tf.reduce_mean(tf.square(returns - vals))

    # Compute gradients
    agent_grads = tape.gradient(actor_loss, agent.trainable_variables)
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)

    # Apply updates
    optimizer_agent.apply_gradients(zip(agent_grads, agent.trainable_variables))
    optimizer_critic.apply_gradients(zip(critic_grads, critic.trainable_variables))

    del tape

    return actor_loss, critic_loss, entropy


def train_agent(agent, critic, optimizer_agent, optimizer_critic,
                features, messages, symbols, preds,
                old_joint_logps, advantages, returns,
                minibatch_size, num_epochs):

    
    batch_size = tf.shape(symbols)[0]
    total_timesteps = tf.shape(symbols)[1]
    buffer_size = int(batch_size * total_timesteps)

    # flattening for 
    def flatten(x):
        x_shape = tf.shape(x)
        return tf.reshape(x, tf.concat([[x_shape[0] * x_shape[1]], x_shape[2:]], axis=0))

    features_flat = tf.repeat(features, repeats=total_timesteps, axis=0) # (batch_size*total_timesteps, num_obj, feature dim)
    messages_flat = flatten(messages)                                    # (batch_size*total_timesteps,)
    symbols_flat = flatten(symbols)                                      # (batch_size*total_timesteps,)
    preds_flat = flatten(preds)                                          # (batch_size*total_timesteps, num_obj)
    old_joint_logps_flat = flatten(old_joint_logps)                      # (batch_size*total_timesteps,)
    advantages_flat = flatten(advantages)                                # (batch_size*total_timesteps,)
    returns_flat = flatten(returns)                                      # (batch_size*total_timesteps,)

    dataset = tf.data.Dataset.from_tensor_slices(
        (features_flat, messages_flat, symbols_flat, preds_flat,
        old_joint_logps_flat, advantages_flat, returns_flat)
    ).shuffle(buffer_size).batch(minibatch_size)

    all_actor_losses, all_critic_losses, all_entropies = [], [], []
    for _ in range(num_epochs):
        for (mb_feats, mb_messages, mb_symbols, mb_preds,
             mb_old_joint_logps, mb_advs, mb_returns) in dataset:


            # dataset entry ist dictionary
            # make sure indices are correct / -> more states than wahrscheinlichkeiten
            # more messages than wahrscheinlichkeiten 

            actor_loss, critic_loss, entropy = train_step(
                agent, critic, optimizer_agent, optimizer_critic,
                mb_feats, mb_messages,
                mb_symbols, mb_preds,
                mb_old_joint_logps, mb_advs, mb_returns
            )
            all_actor_losses.append(actor_loss.numpy())
            all_critic_losses.append(critic_loss.numpy())
            all_entropies.append(entropy.numpy())
            
    return np.mean(all_actor_losses), np.mean(all_critic_losses), np.mean(all_entropies)
    

agent_1 = agents.AgentDummy()
agent_2 = agents.AgentDummy()

critic_1 = agents.AgentDummyCritic()
critic_2 = agents.AgentDummyCritic()

optimizer_agent_1 = tf.keras.optimizers.Adam(1e-2) #1e-3
optimizer_agent_2 = tf.keras.optimizers.Adam(1e-2) #1e-3
optimizer_crit_1 = tf.keras.optimizers.Adam(1e-2) #1e-3
optimizer_crit_2 = tf.keras.optimizers.Adam(1e-2) #1e-3


def train(num_iterations=1000, batch_size=2048, minibatch_size=64, num_epochs=4):


    dataset = generate_dataset(num_same=3, num_diff1=2, num_diff2=2, shuffle_buffer_size=1000, prefetch_buffer_size=1000, batch_size=batch_size, which='TRAIN')


    # create the rollout 
    for iter in range(num_iterations): 

        current_ts = 0
        total_A1_loss = 0.0
        total_A1_crit_loss = 0.0
        total_A2_loss = 0.0
        total_A2_crit_loss = 0.0

        total_minibatches = 0

        max_steps = 10

        # tensor array setup for agent 1
        ta_messages_to_a1 = tf.TensorArray(tf.int32, size=max_steps)
        ta_a1_symbols = tf.TensorArray(tf.int32, size=max_steps)
        ta_a1_preds = tf.TensorArray(tf.float32, size=max_steps)
        ta_a1_img_logps = tf.TensorArray(tf.float32, size=max_steps)
        ta_a1_msg_logps = tf.TensorArray(tf.float32, size=max_steps)
        ta_a1_joint_logps = tf.TensorArray(tf.float32, size=max_steps)
        ta_a1_rewards = tf.TensorArray(tf.float32, size=max_steps)
        ta_a1_vals = tf.TensorArray(tf.float32, size=max_steps)

        # tensor array setup for agent 2
        ta_messages_to_a2 = tf.TensorArray(tf.int32, size=max_steps)
        ta_a2_symbols = tf.TensorArray(tf.int32, size=max_steps)
        ta_a2_preds = tf.TensorArray(tf.float32, size=max_steps)
        ta_a2_img_logps = tf.TensorArray(tf.float32, size=max_steps)
        ta_a2_msg_logps = tf.TensorArray(tf.float32, size=max_steps)
        ta_a2_joint_logps = tf.TensorArray(tf.float32, size=max_steps)
        ta_a2_rewards = tf.TensorArray(tf.float32, size=max_steps)
        ta_a2_vals = tf.TensorArray(tf.float32, size=max_steps)

        for (a1_feats, a1_targets), (a2_feats, a2_targets) in dataset:

            for current_ts in tf.range(max_steps):

                message_to_a1 = tf.cond(
                    tf.equal(current_ts, 0),
                    lambda: tf.zeros([batch_size], dtype=tf.int32),  # initial messsage just filled with zeros
                    lambda: ta_a2_symbols.read(current_ts-1)
                )

                message_to_a2 = tf.cond(
                    tf.equal(current_ts, 0),
                    lambda: tf.zeros([batch_size], dtype=tf.int32),  # initial messsage just filled with zeros
                    lambda: ta_a1_symbols.read(current_ts-1)
                )
                
                ta_messages_to_a1 = ta_messages_to_a1.write(current_ts, message_to_a1)
                ta_messages_to_a2 = ta_messages_to_a2.write(current_ts, message_to_a2)
                
                a1_symbols, a1_preds, a1_img_logps, a1_msg_logps, a1_joint_logps, a1_rewards, a1_vals = rollout(agent_1, critic_1, a1_feats, a1_targets, message_to_a1)
                a2_symbols, a2_preds, a2_img_logps, a2_msg_logps, a2_joint_logps, a2_rewards, a2_vals = rollout(agent_2, critic_2, a2_feats, a2_targets, message_to_a2)

                
                # initialize optimizer slots
                zero_grads_agent_1 = [tf.zeros_like(v) for v in agent_1.trainable_variables]
                zero_grads_critic_1 = [tf.zeros_like(v) for v in critic_1.trainable_variables]
                optimizer_agent_1.apply_gradients(zip(zero_grads_agent_1, agent_1.trainable_variables))
                optimizer_crit_1.apply_gradients(zip(zero_grads_critic_1, critic_1.trainable_variables))
                
                zero_grads_agent_2 = [tf.zeros_like(v) for v in agent_2.trainable_variables]
                zero_grads_critic_2 = [tf.zeros_like(v) for v in critic_2.trainable_variables]
                optimizer_agent_2.apply_gradients(zip(zero_grads_agent_2, agent_2.trainable_variables))
                optimizer_crit_2.apply_gradients(zip(zero_grads_critic_2, critic_2.trainable_variables))

                ta_a1_symbols = ta_a1_symbols.write(current_ts, a1_symbols)
                ta_a1_preds = ta_a1_preds.write(current_ts, a1_preds)
                ta_a1_img_logps = ta_a1_img_logps.write(current_ts, a1_img_logps)
                ta_a1_msg_logps = ta_a1_msg_logps.write(current_ts, a1_msg_logps)
                ta_a1_joint_logps = ta_a1_joint_logps.write(current_ts, a1_joint_logps)
                ta_a1_rewards = ta_a1_rewards.write(current_ts, a1_rewards)
                ta_a1_vals = ta_a1_vals.write(current_ts, a1_vals)

                ta_a2_symbols = ta_a2_symbols.write(current_ts, a2_symbols)
                ta_a2_preds = ta_a2_preds.write(current_ts, a2_preds)
                ta_a2_img_logps = ta_a2_img_logps.write(current_ts, a2_img_logps)
                ta_a2_msg_logps = ta_a2_msg_logps.write(current_ts, a2_msg_logps)
                ta_a2_joint_logps = ta_a2_joint_logps.write(current_ts, a2_joint_logps)
                ta_a2_rewards = ta_a2_rewards.write(current_ts, a2_rewards)
                ta_a2_vals = ta_a2_vals.write(current_ts, a2_vals)
                

        # stacking array from rollout - Agent 1    
        all_messages_to_a1 =ta_messages_to_a1.stack()   # shape: (total_timesteps, batch_size)
        all_a1_symbols = ta_a1_symbols.stack()          # shape: (total_timesteps, batch_size)
        all_a1_preds = ta_a1_preds.stack()              # shape: (total_timesteps, batch_size, num_obj)
        all_a1_img_logps = ta_a1_img_logps.stack()      # shape: (total_timesteps, batch_size)
        all_a1_msg_logps = ta_a1_msg_logps.stack()      # shape: (total_timesteps, batch_size, num_obj)
        all_a1_joint_logps = ta_a1_joint_logps.stack()  # shape: (total_timesteps, batch_size)
        all_a1_rewards = ta_a1_rewards.stack()          # shape: (total_timesteps, batch_size)
        all_a1_vals = ta_a1_vals.stack()                # shape: (total_timesteps, batch_size)

        # transposing to restore batch_size as first 
        all_messages_to_a1 = tf.transpose(all_messages_to_a1, [1,0])    # shape: (batch_size, total_timesteps)
        all_a1_symbols = tf.transpose(all_a1_symbols, [1,0])            # shape: (batch_size, total_timesteps)
        all_a1_preds = tf.transpose(all_a1_preds, [1,0,2])              # shape: (batch_size, total_timesteps, num_obj)
        all_a1_img_logps = tf.transpose(all_a1_img_logps, [1,0])        # shape: (batch_size, total_timesteps)
        all_a1_msg_logps = tf.transpose(all_a1_msg_logps, [1,0,2])      # shape: (batch_size, total_timesteps, num_obj)
        all_a1_joint_logps = tf.transpose(all_a1_joint_logps, [1,0])    # shape: (batch_size, total_timesteps)
        all_a1_rewards = tf.transpose(all_a1_rewards, [1,0])            # shape: (batch_size, total_timesteps)
        all_a1_vals = tf.transpose(all_a1_vals, [1,0])                  # shape: (batch_size, total_timesteps)

        # print("symbols shape after stack: ", all_a1_symbols.shape)
        # print("preds shape after stack: ", all_a1_preds.shape)
        # print(" img_logps shape after stack: ", all_a1_img_logps.shape)
        # print(" msg_logps shape after stack: ", all_a1_msg_logps.shape)
        # print(" joint_logps shape after stack: ", all_a1_joint_logps.shape)
        # print("rewards shape after stack: ", all_a1_rewards.shape)
        # print("vals shape after stack: ", all_a1_vals.shape)
        

        # stacking array from rollout - Agent 2 
        all_messages_to_a2 =ta_messages_to_a2.stack()  
        all_a2_symbols = ta_a2_symbols.stack()      
        all_a2_preds = ta_a2_preds.stack()
        all_a2_img_logps = ta_a2_img_logps.stack()
        all_a2_msg_logps = ta_a2_msg_logps.stack()
        all_a2_joint_logps = ta_a2_joint_logps.stack()
        all_a2_rewards = ta_a2_rewards.stack()     
        all_a2_vals = ta_a2_vals.stack()  

        # transposing to restore batch_size as first 
        all_messages_to_a2 = tf.transpose(all_messages_to_a2, [1,0])   
        all_a2_symbols = tf.transpose(all_a2_symbols, [1,0])
        all_a2_preds = tf.transpose(all_a2_preds, [1,0,2])
        all_a2_img_logps = tf.transpose(all_a2_img_logps, [1,0])
        all_a2_msg_logps = tf.transpose(all_a2_msg_logps, [1,0,2])
        all_a2_joint_logps = tf.transpose(all_a2_joint_logps, [1,0])
        all_a2_rewards = tf.transpose(all_a2_rewards, [1,0])  
        all_a2_vals = tf.transpose(all_a2_vals, [1,0])


        # print("symbols shape after transpose: ", all_a2_symbols.shape)
        # print("preds shape after stack: ", all_a2_preds.shape)
        # print("img_logps shape after stack: ", all_a2_img_logps.shape)
        # print("msg_logps shape after stack: ", all_a2_msg_logps.shape)
        # print("joint_logps shape after stack: ", all_a2_joint_logps.shape)
        # print("rewards shape after stack: ", all_a2_rewards.shape)
        # print("vals shape after stack: ", all_a2_vals.shape)
        
        # print(message_to_a1)

        last_val_a1 = critic_1(a1_feats, message_to_a1)
        all_a1_vals = tf.concat([all_a1_vals, last_val_a1[:, None]], axis=1)  # (batch_size, total_timesteps+1)
        returns_a1, advantages_a1 = compute_gae(all_a1_rewards, all_a1_vals)

        last_val_a2 = critic_2(a2_feats, message_to_a2)
        all_a2_vals = tf.concat([all_a2_vals, last_val_a2[:, None]], axis=1)  # (batch_size, total_timesteps+1)
        returns_a2, advantages_a2 = compute_gae(all_a2_rewards, all_a2_vals)
            
            
        losses_a1 = train_agent(agent_1, critic_1, optimizer_agent_1, optimizer_crit_1,
                        a1_feats, all_messages_to_a1,
                        all_a1_symbols, all_a1_preds,
                        all_a1_joint_logps, advantages_a1, returns_a1,
                        minibatch_size, num_epochs)

        losses_a2 = train_agent(agent_2, critic_2, optimizer_agent_2, optimizer_crit_2,
                                a2_feats, all_messages_to_a2,
                                all_a2_symbols, all_a2_preds,
                                all_a2_joint_logps, advantages_a2, returns_a2,
                                minibatch_size, num_epochs)

        
        print("Losses: ", losses_a1, losses_a2)

        
    
train(num_iterations=2, batch_size=2, minibatch_size=1, num_epochs=1)
        