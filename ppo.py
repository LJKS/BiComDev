import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import agents 
import data 

# @tf.function
# def train_step(target_feats, distractor_feats, input_left, input_right, symbols, responses, sender_logps, receiver_logps,  sender_advantages, receiver_advantages, rewards):    
#     clip_epsilon = 0.2

#     with tf.GradientTape(persistent=True) as tape:
#         sender_probs = sender(target_feats, distractor_feats)
#         sender_probs = tf.squeeze(sender_probs, axis=1) if sender_probs.shape.rank == 3 else sender_probs

#         sender_vals = sender_crit(target_feats, distractor_feats)
#         sender_dist = tfp.distributions.Categorical(probs=sender_probs)
   
#         new_sender_logps = sender_dist.log_prob(symbols)

#         sender_ratio = tf.exp(new_sender_logps - sender_logps)
#         sender_clip = tf.clip_by_value(sender_ratio, 1 - clip_epsilon, 1 + clip_epsilon)
#         sender_entropy = tf.reduce_mean(sender_dist.entropy())

#         receiver_probs = receiver(input_left, input_right, symbols)
#         receiver_vals = receiver_crit(input_left, input_right, symbols)
#         receiver_dist = tfp.distributions.Categorical(probs=receiver_probs)

#         new_receiver_logps = receiver_dist.log_prob(responses)

#         receiver_ratio = tf.exp(new_receiver_logps - receiver_logps)
#         receiver_clip = tf.clip_by_value(receiver_ratio, 1 - clip_epsilon, 1 + clip_epsilon)
#         receiver_entropy = tf.reduce_mean(receiver_dist.entropy())

#         sender_loss = -tf.reduce_mean(tf.minimum(sender_ratio * sender_advantages, sender_clip * sender_advantages)) - 0.01 * sender_entropy
#         sender_crit_loss = tf.reduce_mean(tf.square(sender_advantages - sender_vals)) # tf.squeeze(sender_vals)
        
#         receiver_loss = -tf.reduce_mean(tf.minimum(receiver_ratio * receiver_advantages, receiver_clip * receiver_advantages))  - 0.01 * receiver_entropy
#         receiver_crit_loss = tf.reduce_mean(tf.square(receiver_advantages - receiver_vals)) #tf.squeeze(receiver_vals)

#     sender_grads =  tape.gradient(sender_loss, sender.trainable_variables)
#     sender_crit_grads = tape.gradient(sender_crit_loss, sender_crit.trainable_variables)
#     receiver_grads = tape.gradient(receiver_loss, receiver.trainable_variables)
#     receiver_crit_grads = tape.gradient(receiver_crit_loss, receiver_crit.trainable_variables)

#     optimizer_sender.apply_gradients(zip(sender_grads, sender.trainable_variables))
#     optimizer_sender_crit.apply_gradients(zip(sender_crit_grads, sender_crit.trainable_variables))
#     optimizer_receiver.apply_gradients(zip(receiver_grads, receiver.trainable_variables))
#     optimizer_receiver_crit.apply_gradients(zip(receiver_crit_grads, receiver_crit.trainable_variables))

#     return {
#         "sender_loss": sender_loss,
#         "sender_crit_loss": sender_crit_loss,
#         "receiver_loss": receiver_loss,
#         "receiver_crit_loss": receiver_crit_loss
#     }
def sender(agent,critic, features):
    sender_probs = agent(features, role=tf.constant(0)) # output shape: (batch_size, vocab_size)
    sender_vals = critic(features,role=tf.constant(0)) # output shape: (batch,)

    sender_dist = tfp.distributions.Categorical(probs=sender_probs)
    symbols = sender_dist.sample()       
    symbols = tf.cast(symbols, tf.int32) 
    sender_logps = sender_dist.log_prob(symbols)
    return sender_logps, symbols, sender_vals

def receiver(agent, critic, features, mask, message):
    receiver_probs = agent(features, role=tf.constant(1), input_message=message)
    receiver_vals = critic(features, role=tf.constant(1), input_message=message)

    pred = tf.cast(receiver_probs > 0.5, tf.int32)

    # normalized by target_num to avoid reward inflation from higher number of targets
    target_mask = tf.cast(tf.equal(mask, 0), tf.float32)
    num_targets = tf.reduce_sum(target_mask, axis=-1)
    reward = tf.reduce_sum(receiver_probs * target_mask, axis=-1) / (num_targets + 1e-8)

    return pred, reward, receiver_vals


def combined(agent, critic, features, mask, message):
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
    target_mask = tf.cast(tf.equal(mask, 0), tf.float32)
    num_targets = tf.reduce_sum(target_mask, axis=-1) + 1e-8
    correct = tf.reduce_sum(preds * target_mask, axis=-1)  
    rewards = correct / (num_targets + 1e-8)       

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


def train(num_iterations=1000, batch_size=2048, minibatch_size=64, num_epochs=4):

    # load the data
    # train_ds, val_ds, test_ds = data.load_coco_captions(data_dir="./data")

    agent_1 = agents.AgentDummy()
    agent_2 = agents.AgentDummy()

    critic_1 = agents.AgentDummyCritic()
    critic_2 = agents.AgentDummyCritic()

    optimizer_agent_1 = tf.keras.optimizers.Adam(1e-2) #1e-3
    optimizer_agent_2 = tf.keras.optimizers.Adam(1e-2) #1e-3
    optimizer_crit_1 = tf.keras.optimizers.Adam(1e-2) #1e-3
    optimizer_crit_2 = tf.keras.optimizers.Adam(1e-2) #1e-3


    done = False

    # create the rollout 
    for iter in range(num_iterations): 

        current_ts = 0
        total_A1_loss = 0.0
        total_A1_crit_loss = 0.0
        total_A2_loss = 0.0
        total_A2_crit_loss = 0.0

        total_minibatches = 0

        
        max_steps = 10

        # all_a1_symbols, all_a1_preds, all_a1_img_logps, all_a1_msg_logps, all_a1_joint_logps, all_a1_rewards, all_a1_vals = [],[],[],[],[],[],[]
        # all_a2_symbols, all_a2_preds, all_a2_img_logps, all_a2_msg_logps, all_a2_joint_logps, all_a2_rewards, all_a2_vals = [],[],[],[],[],[],[]
        
        # tensor array setup for agent 1
        ta_a1_symbols = tf.TensorArray(tf.int32, size=max_steps)
        ta_a1_preds = tf.TensorArray(tf.float32, size=max_steps)
        ta_a1_img_logps = tf.TensorArray(tf.float32, size=max_steps)
        ta_a1_msg_logps = tf.TensorArray(tf.float32, size=max_steps)
        ta_a1_joint_logps = tf.TensorArray(tf.float32, size=max_steps)
        ta_a1_rewards = tf.TensorArray(tf.float32, size=max_steps)
        ta_a1_vals = tf.TensorArray(tf.float32, size=max_steps)

        # tensor array setup for agent 2
        ta_a2_symbols = tf.TensorArray(tf.int32, size=max_steps)
        ta_a2_preds = tf.TensorArray(tf.float32, size=max_steps)
        ta_a2_img_logps = tf.TensorArray(tf.float32, size=max_steps)
        ta_a2_msg_logps = tf.TensorArray(tf.float32, size=max_steps)
        ta_a2_joint_logps = tf.TensorArray(tf.float32, size=max_steps)
        ta_a2_rewards = tf.TensorArray(tf.float32, size=max_steps)
        ta_a2_vals = tf.TensorArray(tf.float32, size=max_steps)

        batch_size = 12
        num_obj = 15

        # images = data.sample_and_embed_img(train_ds, num_img=6, batch_size=5)
        numbers = data.create_dummy_data(num_obj=num_obj, batch_size=batch_size)

        feats_agent1, feats_agent2, assignment_mask = data.assign_feats_to_agents(numbers, num_same=1, num_diff1=7, num_diff2=7) # feats: shape=(batch_size, num_img, 2048), dtype=float32
        feats_agent1_shuffled, shuffled_mask1= data.shuffle_features_and_targets(feats_agent1, assignment_mask)
        feats_agent2_shuffled, shuffled_mask2 = data.shuffle_features_and_targets(feats_agent2, assignment_mask)
        # print(feats_agent1_shuffled)

        # initial message for both agents
        # message_to_a1 = tf.zeros([batch_size], dtype=tf.int32)
        # message_to_a2 = tf.zeros([batch_size], dtype=tf.int32)

        for current_ts in tf.range(max_steps):

            message_to_a1 = tf.cond(
                tf.equal(current_ts, 0),
                lambda: tf.zeros([batch_size], dtype=tf.int32),  
                lambda: ta_a2_symbols.read(current_ts-1)
            )

            message_to_a2 = tf.cond(
                tf.equal(current_ts, 0),
                lambda: tf.zeros([batch_size], dtype=tf.int32),  
                lambda: ta_a1_symbols.read(current_ts-1)
            )
            print(message_to_a1)
            
            a1_symbols, a1_preds, a1_img_logps, a1_msg_logps, a1_joint_logps, a1_rewards, a1_vals = combined(agent_1, critic_1, feats_agent1_shuffled, shuffled_mask1, message_to_a1)
            a2_symbols, a2_preds, a2_img_logps, a2_msg_logps, a2_joint_logps, a2_rewards, a2_vals= combined(agent_2, critic_2, feats_agent2_shuffled, shuffled_mask2, message_to_a2)

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
        all_a1_symbols = ta_a1_symbols.stack()          # shape: (total_timesteps, batch_size)
        all_a1_preds = ta_a1_preds.stack()              # shape: (total_timesteps, batch_size, num_obj)
        all_a1_img_logps = ta_a1_img_logps.stack()      # shape: (total_timesteps, batch_size)
        all_a1_msg_logps = ta_a1_msg_logps.stack()      # shape: (total_timesteps, batch_size, num_obj)
        all_a1_joint_logps = ta_a1_joint_logps.stack()  # shape: (total_timesteps, batch_size)
        all_a1_rewards = ta_a1_rewards.stack()          # shape: (total_timesteps, batch_size)
        all_a1_vals = ta_a1_vals.stack()                # shape: (total_timesteps, batch_size)

        # transposing to restore batch_size as first 
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
        all_a2_symbols = ta_a2_symbols.stack()      
        all_a2_preds = ta_a2_preds.stack()
        all_a2_img_logps = ta_a2_img_logps.stack()
        all_a2_msg_logps = ta_a2_msg_logps.stack()
        all_a2_joint_logps = ta_a2_joint_logps.stack()
        all_a2_rewards = ta_a2_rewards.stack()     
        all_a2_vals = ta_a2_vals.stack()  


        # transposing to restore batch_size as first 
        all_a2_symbols = tf.transpose(all_a2_symbols, [1,0])
        all_a2_preds = tf.transpose(all_a2_preds, [1,0,2])
        all_a2_img_logps = tf.transpose(all_a2_img_logps, [1,0])
        all_a2_msg_logps = tf.transpose(all_a2_msg_logps, [1,0,2])
        all_a2_joint_logps = tf.transpose(all_a2_joint_logps, [1,0])
        all_a2_rewards = tf.transpose(all_a2_rewards, [1,0])  
        all_a2_vals = tf.transpose(all_a2_vals, [1,0])


        # print("symbols shape after stack: ", all_a2_symbols.shape)
        # print("preds shape after stack: ", all_a2_preds.shape)
        # print(" img_logps shape after stack: ", all_a2_img_logps.shape)
        # print(" msg_logps shape after stack: ", all_a2_msg_logps.shape)
        # print(" joint_logps shape after stack: ", all_a2_joint_logps.shape)
        # print("rewards shape after stack: ", all_a2_rewards.shape)
        # print("vals shape after stack: ", all_a2_vals.shape)
        


        # final_state = feats_agent1_shuffled[:, -1, :]  # (B, feature_dim)
        # print(final_state)
        # last_message_to_a1 = all_a2_symbols[:,-1]
        # print(last_message_to_a1)
        # last_val = critic_1(final_state,last_message_to_a1)
        # all_a1_vals = tf.concat([all_a1_vals, last_val[:, None]], axis=1)  # (B, T+1)
        # returns_a1, advs_a1 = compute_gae(all_a1_rewards, all_a1_vals)
        # print("returns_a1: ", returns_a1)
        # print("Advantages a1: ", advs_a1)

        break
            
        
            # print("Advantages: ", agent1_advantages)

            # dataset = tf.data.Dataset.from_tensor_slices((target_feats, distractor_feats, input_left, input_right, symbols, responses, sender_logps, receiver_logps, sender_advantages, receiver_advantages, rewards))
            # dataset = dataset.shuffle(batch_size).batch(minibatch_size)

            # for _ in range(num_epochs):
            #     for mb_target, mb_distractor, mb_left, mb_right, mb_symbols, mb_responses, mb_sender_logp, mb_receiver_logp, mb_sender_advantages, mb_receiver_advantages, mb_rewards in dataset:
            #         losses = train_step(mb_target, mb_distractor, 
            #                    mb_left, mb_right, 
            #                    mb_symbols, mb_responses,
            #                    mb_sender_logp, mb_receiver_logp, 
            #                    mb_sender_advantages, mb_receiver_advantages,
            #                    mb_rewards)
                    
            #         total_sender_loss += losses["sender_loss"].numpy()
            #         total_sender_crit_loss += losses["sender_crit_loss"].numpy()
            #         total_receiver_loss += losses["receiver_loss"].numpy()
            #         total_receiver_crit_loss += losses["receiver_crit_loss"].numpy()
            #         total_minibatches += 1
            
            # avg_reward = np.mean(rewards_list)
            # print(f"Iter {iter} | "
            #     f"Reward: {avg_reward:.3f} | "
            #     f"S_loss: {total_sender_loss/total_minibatches:.4f} | "
            #     f"S_vloss: {total_sender_crit_loss/total_minibatches:.4f} | "
            #     f"R_loss: {total_receiver_loss/total_minibatches:.4f} | "
            #     f"R_vloss: {total_receiver_crit_loss/total_minibatches:.4f}")
                
        

train(num_iterations=10, batch_size=1, minibatch_size=1, num_epochs=1)
        
