import tensorflow as tf

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """computes the generalized advantage estimation based on critics value estimate and rewards
    
    Args:
        rewards (tensor): rewards collected from a (game) rollout  
        values (tensor): critics value estimate collected from a (game) rollout
        gamma (float): discount factor for future rewards
        lam (float): bias-variance tradeoff parameter

    Return:
        Tensors containing the advantages and returns following the gae
        """
    batch_size = tf.shape(rewards)[0]
    total_timesteps = tf.shape(rewards)[1]

    advs = [] 
    gae = tf.zeros((batch_size,), dtype=tf.float32)
    

    for t in reversed(range(total_timesteps)): # loop backwards
        delta = rewards[:, t] + gamma * values[:, t+1] - values[:, t] 
        gae = delta + gamma * lam * gae
        advs.append(gae)

    advs = tf.stack(advs[::-1], axis=1) # stack in reversed order 
    returns = advs + values[:, :-1]  

    return returns, advs


def target_match_ratio(preds, targets):
    """a reward function that """
    num_targets = tf.reduce_sum(targets)
    correct = tf.reduce_sum(preds * targets, axis=-1)  
    rewards = correct / num_targets
    return rewards



def initialize_optimizer_slots(optimizer, model):
    zero_grads = [tf.zeros_like(v) for v in model.trainable_variables]
    optimizer.apply_gradients(zip(zero_grads, model.trainable_variables))

