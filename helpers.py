import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
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
        rewards optained by comparing prediction and targets
    """
    # preds = tf.cast(preds, tf.float32)
    # targets = tf.cast(targets, tf.float32)

    correct = tf.cast(tf.equal(preds, targets), dtype=tf.float32)

    rewards = correct / targets.shape[1]
    rewards = tf.reduce_sum(rewards, axis=-1)
    return rewards


# def target_match_ratio(preds, targets):
#     """a reward function that measures correctness of agent predictions against the target set
#     Args: 
#         preds (tensor): the agent's prediction on what images are targets 
#         targets (tensor): A binary tensor that details whether a feature tensor is a target (=1) or not (=0); shape [batch_size, num_img]
    
#     Returns:
#         rewards optained by
#     """
#     num_targets = tf.reduce_sum(targets, axis=-1)[0]
#     correct = (preds * targets)

#     rewards = correct / num_targets 
#     rewards = tf.reduce_sum(rewards, axis=-1)
#     return rewards


def calcualte_train_reward(merged_rollouts):
    r1 = tf.reduce_mean(merged_rollouts["agent_1"]["rewards"])
    r2 = tf.reduce_mean(merged_rollouts["agent_2"]["rewards"])
    mean_reward = 0.5 * (r1 + r2)
    
    return mean_reward


def calculate_train_accuracy(merged_rollouts):
    accs = []
    for agent in ["agent_1", "agent_2"]:
        preds = merged_rollouts[agent]["preds"]     
        targets = merged_rollouts[agent]["targets"] 

        correct = tf.cast(tf.equal(preds, targets), tf.float32)
        acc = tf.reduce_mean(correct) 
        accs.append(acc)

    return 0.5 * (accs[0] + accs[1])


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


def visualize_training_curves(training_accuracies, training_rewards, eval_rewards, actor_losses, critic_losses, save_path=None):
    """plots the average training reward, average evaluation reward, and actor and critic losses, all per epoch
    Args:
        training_rewards (list): A list that contains per-step reward, aggregated and logged once per epoch from rollout data
        eval_rewards (list): 
        actor_losses ():
        critic_losses ():

    """
    fig, axs = plt.subplots(5, 1, figsize=(8, 10))

    axs[0].plot(training_accuracies, marker='o')
    axs[0].set_title("Mean Accuracies per Epoch")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")

    axs[1].plot(training_rewards, marker='o')
    axs[1].set_title("Mean Training Reward per Epoch")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Reward")
    
    axs[2].plot(eval_rewards, marker='o')
    axs[2].set_title("Mean Evaluation Reward per Epoch")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Reward")

    axs[3].plot(actor_losses, marker='x')
    axs[3].set_title("Mean Actor Loss ")
    axs[3].set_xlabel("Epoch")
    axs[3].set_ylabel("Loss")

    axs[4].plot(critic_losses, marker='x')
    axs[4].set_title("Mean Critic Loss")
    axs[4].set_xlabel("Epoch")
    axs[4].set_ylabel("Loss")

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

def save_results(run_dir, results):
    NotImplemented


def save_step_metrics(results, run_dir, filename="training_steps.csv"):
    df = pd.DataFrame({
        "step": range(len(results["raw_actor_losses"])),
        "actor_loss": results["raw_actor_losses"],
        "critic_loss": results["raw_critic_losses"],
    })
    path = os.path.join(run_dir, filename)
    df.to_csv(path, index=False)

def save_epoch_metrics(results, run_dir,  filename="training_epochs.csv"):
    num_epochs = len(results["mean_actor_losses_per_epoch"])

    df = pd.DataFrame({
        "epoch": range(num_epochs),
        "mean_actor_loss": results["mean_actor_losses_per_epoch"],
        "mean_critic_loss": results["mean_critic_losses_per_epoch"],
        "train_accuracies": results["train_accuracies"],
        "train_reward": results["train_rewards"],
        "eval_reward": results["eval_rewards"],
    })
    path = os.path.join(run_dir, filename)
    df.to_csv(path, index=False)


import pickle


def save_raw_data(results, run_dir, filename="raw_data.pkl"):

    data_to_save = {
        "raw_rewards": results["raw_rewards"],
        "preds": results["preds"]
    }
    
    path = os.path.join(run_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Raw rewards and predictions saved to {path}")
