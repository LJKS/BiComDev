import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
import json


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """computes the generalized advantage estimation based on critics value estimate and rewards
    
    Args:
        rewards (tensor): rewards collected from a (game) rollout  
        values (tensor): critics value estimate collected from a (game) rollout
        gamma (float): discount factor for future rewards
        lam (float): bias-variance tradeoff parameter (lambda)

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
    """a reward function that measures correctness of agent predictions against the target set
    Args: 
        preds (tensor): the agent's prediction on what images are targets 
        targets (tensor): A binary tensor that details whether a feature tensor is a target (=1) or not (=0); shape [batch_size, num_img]
    
    Returns:
        rewards optained by
    """
    num_targets = tf.reduce_sum(targets)
    correct = tf.reduce_sum(preds * targets, axis=-1)  
    rewards = correct / num_targets
    return rewards


def combine_dicts(agent_1_dict, agent_2_dict):
    '''removes agent branching logic to pass all collected data for ppo update'''
    return {
        key: tf.concat([agent_1_dict[key], agent_2_dict[key]], axis=0)
        for key in agent_1_dict.keys()
    }



def initialize_optimizer_slots(optimizer, model):
    """function that intialises the optimizer slots to not run into a singleton error"""
    zero_grads = [tf.zeros_like(v) for v in model.trainable_variables]
    optimizer.apply_gradients(zip(zero_grads, model.trainable_variables))


def calcualte_train_reward(merged_rollouts):
    r1 = tf.reduce_mean(merged_rollouts["agent_1"]["rewards"])
    r2 = tf.reduce_mean(merged_rollouts["agent_2"]["rewards"])
    mean_reward = 0.5 * (r1 + r2)
    
    return mean_reward


def visualize_training_curves(training_rewards, eval_rewards, actor_losses, critic_losses, save_path=None):
    """plots the average training reward, average evaluation reward, and actor and critic losses, all per epoch
    Args:
        training_rewards (list): A list that contains per-step reward, aggregated and logged once per epoch from rollout data
        eval_rewards (list): 
        actor_losses ():
        critic_losses ():

    """
    fig, axs = plt.subplots(4, 1, figsize=(8, 10))

    axs[0].plot(training_rewards, marker='o')
    axs[0].set_title("Mean Training Reward per Epoch")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Reward")
    
    axs[1].plot(eval_rewards, marker='o')
    axs[1].set_title("Mean Evaluation Reward per Epoch")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Reward")

    axs[2].plot(actor_losses, marker='x')
    axs[2].set_title("Mean Actor Loss ")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Loss")

    axs[3].plot(critic_losses, marker='x')
    axs[3].set_title("Mean Critic Loss")
    axs[3].set_xlabel("Epoch")
    axs[3].set_ylabel("Loss")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # create folder if it doesn't exist
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    # plt.show()



def setup_run_dir(args):

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("runs", run_id)

    os.makedirs(run_dir, exist_ok=False)
    os.makedirs(os.path.join(run_dir, "checkpoints"))
    os.makedirs(os.path.join(run_dir, "plots"))

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    return run_dir

def save_checkpoints(run_dir, current_epoch, agent, critic, num_epochs, save_every):
    
    if current_epoch % save_every != 0 and current_epoch != (num_epochs-1):
            return  # skip saving

    ckpt_dir = os.path.join(run_dir, "checkpoints", f"epoch_{current_epoch:04d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    agent.save_weights(os.path.join(ckpt_dir, "actor.weights.h5"))
    critic.save_weights(os.path.join(ckpt_dir, "critic.weights.h5"))

