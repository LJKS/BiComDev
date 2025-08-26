import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# from agents import AgnosticSender, Receiver, SenderCritic, ReceiverCritic
import data 



# sender = AgnosticSender()
# receiver = Receiver()
# sender_crit = SenderCritic()
# receiver_crit = ReceiverCritic()

# optimizer_sender = tf.keras.optimizers.Adam(1e-2) #1e-3
# optimizer_receiver = tf.keras.optimizers.Adam(1e-2) #1e-3
# optimizer_sender_crit = tf.keras.optimizers.Adam(1e-2) #1e-3
# optimizer_receiver_crit = tf.keras.optimizers.Adam(1e-2) #1e-3

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

def train(num_iterations=1000, batch_size=2048, minibatch_size=64, num_epochs=4):

    # load the data
    train_ds, val_ds, test_ds = data.load_coco_captions(data_dir="./data")

    agent_1 = agents.Agent_1()
    agent_2 = agents.Agent_2()

    done = false
    # create the rollout 
    for iter in range(num_iterations): 
        total_A1_loss = 0.0
        total_A1_crit_loss = 0.0
        total_A2_loss = 0.0
        total_A2_crit_loss = 0.0

        total_minibatches = 0

        rewards_list = []


        target_feats, distractor_feats = [], []

        imgs = data.sample_and_embed_img(train_ds, num_img=6)

        feats_agent1, feats_agent2, assignment_mask = data.assign_feats_to_agents(imgs, batch_size=1, num_img=6, num_same=2, num_diff1=2, num_diff2=2) # feats: shape=(batch_size, num_img, 2048), dtype=float32

        print("Features Agent 1: ", feats_agent1)
        
        while not done:
            sender =  "current sender depending on the timestep"
            receiver = "current receiver depending on the timestep"
            # sender_probs = sender(target_feats, distractor_feats)
            # sender_probs = tf.squeeze(sender_probs, axis=1) if sender_probs.shape.rank == 3 else sender_probs
            # sender_vals = sender_crit(target_feats, distractor_feats)
            # sender_vals = tf.squeeze(sender_vals, axis=-1)
            # sender_dist = tfp.distributions.Categorical(probs=sender_probs)
            # symbols = sender_dist.sample()       
            # symbols = tf.cast(symbols, tf.int32) 
            # sender_logps = sender_dist.log_prob(symbols)
            # # print("Sender_logps", sender_logps.shape)

            # swap = tf.random.uniform((batch_size,)) < 0.5
            # target_feats = tf.squeeze(target_feats) # shape (B, 4096)
            # distractor_feats = tf.squeeze(distractor_feats) # shape (B, 4096)
            # input_left = tf.where(swap[:, tf.newaxis], target_feats, distractor_feats)
            # input_right = tf.where(swap[:, tf.newaxis], distractor_feats, target_feats)
            # correct_choice = tf.where(swap, tf.zeros((batch_size,), dtype=tf.int64),
            #                             tf.ones((batch_size,), dtype=tf.int64))
            # receiver_probs = receiver(input_left, input_right, symbols)
            # receiver_dist = tfp.distributions.Categorical(probs=receiver_probs)

            # receiver_vals = receiver_crit(input_left, input_right, symbols)
            # responses = tf.argmax(receiver_probs, axis=-1, output_type=tf.int64)
            # receiver_logps = receiver_dist.log_prob(responses)
            # rewards = tf.cast(tf.equal(responses, correct_choice), tf.float32)  # (B,)
            # rewards_list.extend(rewards.numpy())

            
            # sender_advantages = rewards - sender_vals
            # receiver_advantages = rewards - receiver_vals

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
        
