import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from datetime import datetime
import json
import pickle


# region calculations (and other functional helpers)

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
    preds = tf.cast(preds, tf.float32)
    targets = tf.cast(targets, tf.float32)

    correct = tf.cast(tf.equal(preds, targets), tf.float32)
    num_img = tf.cast(tf.shape(targets)[1], tf.float32)

    rewards = tf.reduce_sum(correct, axis=-1) / num_img

    return rewards


def calculate_train_reward(merged_rollouts):
    """calculates mean train_reward based onn joint rewards from a full epochs rollout 
    Args:
        merged_rollouts (dict): dictionary that contains k*m*n rollouts from each agent
    Return:
        mean training reward for all steps in merged_rollouts
    """
    # calc based on agent_1 only as both agents have the same joint_rewards
    train_reward = tf.reduce_mean(merged_rollouts["agent_1"]["joint_rewards"])
    return train_reward


def calculate_mean_label_accuracy(merged_rollouts):
    """ Calculates mean accuracies based on one epochs rollout;
        Bit-wise accuracy, represents also partially correct predictions
     Args:
        merged_rollouts (dict): dictionary that contains k*m*n rollouts from each agent
    Return:
        mean label accuracy
    """
    accs = []
    for agent in ["agent_1", "agent_2"]:
        preds = merged_rollouts[agent]["preds"]     
        targets = merged_rollouts[agent]["targets"] 

        correct = tf.cast(tf.equal(preds, targets), tf.float32)
        acc = tf.reduce_mean(correct) 
        accs.append(acc)
    train_accuracy = 0.5 * (accs[0] + accs[1])
    return train_accuracy

def calculate_exact_match_accuracy(merged_rollouts):
    """ Calculates mean accuracies based on one epochs rollout;
        Accuracy, represents only fully correct predictions
     Args:
        merged_rollouts (dict): dictionary that contains k*m*n rollouts from each agent
    Return:
        mean exact match accuracy 
    """
    accs = []
    for agent in ["agent_1", "agent_2"]:
        preds = merged_rollouts[agent]["preds"]
        targets = merged_rollouts[agent]["targets"]
        per_sample = tf.reduce_all(tf.equal(preds, targets), axis=-1)  # all images correct
        accs.append(tf.cast(per_sample, tf.float32))
    exact_match_accuracy = 0.5 * (tf.reduce_mean(accs[0]) + tf.reduce_mean(accs[1]))
    return exact_match_accuracy

def linear_anneal(epoch, start_epoch, end_epoch, start_val, end_val):
    """
    Linearly interpolate from start_val to end_val between start_epoch and end_epoch, constant outside interval
    """
    epoch = tf.cast(epoch, tf.float32)
    start_epoch = tf.cast(start_epoch, tf.float32)
    end_epoch = tf.cast(end_epoch, tf.float32)

    # avoid div by zero if someone sets start==end
    denom = tf.maximum(end_epoch - start_epoch, 1.0)
    t = (epoch - start_epoch) / denom
    t = tf.clip_by_value(t, 0.0, 1.0)
    return tf.cast(start_val, tf.float32) + t * (tf.cast(end_val, tf.float32) - tf.cast(start_val, tf.float32))


def combine_dicts(agent_1_dict, agent_2_dict):
    """removes agent branching logic to pass all collected data for ppo update"""
    return {
        key: tf.concat([agent_1_dict[key], agent_2_dict[key]], axis=0)
        for key in agent_1_dict.keys()
    }

def initialize_optimizer_slots(optimizer, model):
    """function that intialises the optimizer slots to not run into a singleton error"""
    zero_grads = [tf.zeros_like(v) for v in model.trainable_variables]
    optimizer.apply_gradients(zip(zero_grads, model.trainable_variables))

#endregion

# region Saving

def setup_run_dir(args):
    """creates a dictionary to save all collected data from given training run
    Args:
        args (Args.Namespace): The configuration of the current run; contains all externally defineable variables
    """
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("runs", run_id)

    os.makedirs(run_dir, exist_ok=False)
    os.makedirs(os.path.join(run_dir, "checkpoints"))
    os.makedirs(os.path.join(run_dir, "plots"))

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    return run_dir

def save_checkpoints(run_dir, current_epoch, agent, critic, num_epochs, save_every, buffer_data=None):
    """saves network weights and intermediary results in save_every intervals
    Args:
        run_dir (str): The path to the directory of the current run
        current_epoch(int): the current epoch
        agent, critic (tf.keras.Model): The current instances of the actor and critic networks
        num_epochs (int): the max number of epochs of training
        save_every (int): interval in which data will be saved
        buffer_data (dict): dictionary of intermediate results (mostly collected for debug runs that might have terminated early)
    """
    if current_epoch % save_every != 0 and current_epoch != (num_epochs-1):
            return  # skip saving

    ckpt_dir = os.path.join(run_dir, "checkpoints", f"epoch_{current_epoch:04d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    agent.save_weights(os.path.join(ckpt_dir, "actor.weights.h5"))
    critic.save_weights(os.path.join(ckpt_dir, "critic.weights.h5"))

    # print(f"Epoch {current_epoch} done!")
    # Save intermediate metrics
    if buffer_data is not None:
        save_intermediate_metrics(ckpt_dir, **buffer_data)


def save_step_metrics(results, run_dir, filename="training_steps.csv"):
    """saves the stepwise data collected separately from epoch means and save as CSV file
    Args:
        results (dict): results collected over the whole training
        run_dir (str): The path to the directory of the current run
        file_name (str): Name under which the data will be save; exptected to be CSV format
    """
    df = pd.DataFrame({
        "step": range(len(results["raw_actor_losses"])),
        "actor_loss": results["raw_actor_losses"],
        "critic_loss": results["raw_critic_losses"],
    })
    path = os.path.join(run_dir, filename)
    df.to_csv(path, index=False)

def save_epoch_metrics(results, run_dir,  filename="training_epochs.csv"):
    """saves the mean data per each epoch of full training and save as CSV file
    Args:
        results (dict): results collected over the whole training
        run_dir (str): The path to the directory of the current run
        file_name (str): Name under which the data will be save; exptected to be CSV format
    """

    num_epochs = len(results["mean_actor_losses_per_epoch"])

    df = pd.DataFrame({
        "epoch": range(num_epochs),
        "mean_actor_loss": results["mean_actor_losses_per_epoch"],
        "mean_critic_loss": results["mean_critic_losses_per_epoch"],
        "train_accuracies": results["train_accuracies"],
        "exact_match_accuracies": results["exact_match_accuracies"],
        "train_reward": results["train_rewards"],
    })
    path = os.path.join(run_dir, filename)
    df.to_csv(path, index=False)


def save_raw_data(results, run_dir, filename="raw_data.pkl"):
    """saves per step data of both agents
    Args:

    """
    data_to_save = {
        "raw_rewards": results["raw_rewards"], # joint rewards of each agent, no mean over epoch yet (/= train_rewards)
        "preds": results["preds"],
        "messages": results["messages"],
        "targets": results["targets"],
    }
    
    path = os.path.join(run_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(data_to_save, f)


def save_intermediate_metrics(ckpt_dir,rewards,actor_losses,critic_losses,messages,preds,targets):
    def to_numpy(x):
        if isinstance(x, tf.Tensor):
            return x.numpy()
        return x

    data = {
        "rewards": [to_numpy(r) for r in rewards],
        "actor_losses": [to_numpy(l) for l in actor_losses],
        "critic_losses": [to_numpy(l) for l in critic_losses],
        "messages": [to_numpy(m) for m in messages],
        "predictions": [to_numpy(p) for p in preds],
        "targets": [to_numpy(t) for t in targets],
    }

    with open(os.path.join(ckpt_dir, "intermediate_data.pkl"), "wb") as f:
        pickle.dump(data, f)

def load_intermediate_metrics(run_dir):
    ckpt_root = os.path.join(run_dir, "checkpoints")

    epochs = []
    mean_rewards = []
    actor_loss_means = []
    critic_loss_means = []

    for folder in sorted(os.listdir(ckpt_root)):
        data_path = os.path.join(ckpt_root, folder, "intermediate_data.pkl")
        if not os.path.exists(data_path):
            continue

        epoch_num = int(folder.split("_")[1])

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        rewards = np.array(data["rewards"])
        actor_losses = np.array(data["actor_losses"])
        critic_losses = np.array(data["critic_losses"])

        epochs.append(epoch_num)
        mean_rewards.append(rewards.mean())
        actor_loss_means.append(actor_losses.mean())
        critic_loss_means.append(critic_losses.mean())

    return epochs, mean_rewards, actor_loss_means, critic_loss_means
#endregion


# region Plotting

def visualize_training_curves(training_rewards, training_accuracies, match_accuracies, actor_losses, critic_losses, save_path=None):
    """plots the average training reward, average evaluation reward, and actor and critic losses, all per epoch"""

    fig, axs = plt.subplots(5, 1, figsize=(8, 10))

    axs[0].plot(training_rewards)
    axs[0].set_title("Mean Training Reward per Epoch")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Reward")

    axs[1].plot(training_accuracies)
    axs[1].set_title("Mean Accuracies per Epoch (image-wise)")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Single Image Accuracy")

    axs[2].plot(match_accuracies)
    axs[2].set_title("Mean Accuracies per Epoch (fully correct)")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Exact Match Accuracy")

    axs[3].plot(actor_losses)
    axs[3].set_title("Mean Actor Loss ")
    axs[3].set_xlabel("Epoch")
    axs[3].set_ylabel("Loss")

    axs[4].plot(critic_losses)
    axs[4].set_title("Mean Critic Loss")
    axs[4].set_xlabel("Epoch")
    axs[4].set_ylabel("Loss")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) 
        plt.savefig(save_path)
    else: 
        plt.show()

#endregion
