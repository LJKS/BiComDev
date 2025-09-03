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
    sender_probs = agent(features, role=tf.constant(0)) # output shape: (num_batches, vocab_size)
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

        rewards_list = []

        # images = data.sample_and_embed_img(train_ds, num_img=6, num_batches=5)
        numbers = data.create_dummy_data(num_obj=15, num_batches=12)

        feats_agent1, feats_agent2, assignment_mask = data.assign_feats_to_agents(numbers, num_same=1, num_diff1=7, num_diff2=7) # feats: shape=(num_batches, num_img, 2048), dtype=float32
        feats_agent1_shuffled, shuffled_mask1= data.shuffle_features_and_targets(feats_agent1, assignment_mask)
        feats_agent2_shuffled, shuffled_mask2 = data.shuffle_features_and_targets(feats_agent2, assignment_mask)
        # print(feats_agent1_shuffled)

        while not done:
            # print("current timestep: ", current_ts)
            if current_ts%2==0:
                sender_logps, symbols, sender_vals = sender(agent=agent_1, critic=critic_1, features=feats_agent1_shuffled)
                receiver_preds, rewards, receiver_vals = receiver(agent=agent_2, 
                                                                critic=critic_2, 
                                                                features=feats_agent2_shuffled, 
                                                                mask=shuffled_mask1, 
                                                                message=symbols)
            else:
                sender_logps, symbols, sender_vals = sender(agent=agent_2, critic=critic_2, features=feats_agent2_shuffled)
                receiver_preds, rewards, receiver_vals = receiver(agent=agent_2, 
                                                                critic=critic_2, 
                                                                features=feats_agent2_shuffled, 
                                                                mask=shuffled_mask1, 
                                                                message=symbols)

            print("Sender logps: ", sender_logps)
            print("rewards: ", rewards)

            current_ts+=1
            if current_ts>10:
                break
            
            
            sender_advantages = rewards - sender_vals
            receiver_advantages = rewards - receiver_vals

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
        
