import helpers
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import json
import math
import ppo
import numpy as np
from sklearn.linear_model import LinearRegression

def plot_relative(runs_a: list, runs_desc_a: str, runs_b: list, runs_desc_b: str):
    """
    Plots accuracies vs training epoch for two set of runs.
    args:
    runs_a: list of run directories for set A
    runs_desc_a: description for set A (for legend)
    runs_b: list of run directories for set B
    runs_desc_b: description for set B (for legend)
    """
    def load_run_raw_data(run_dir):
        runs = []
        for run in run_dir:
            runs.append(helpers.load_raw_data(run))
        return runs

    runs_a_data = load_run_raw_data(runs_a)
    runs_b_data = load_run_raw_data(runs_b)

    def get_average_reward_per_epoch(run_data):
        reward_data = run_data['raw_rewards'] #list of epochs, each containing tensor of shape updatesxgame_steps
        average_rewards = []
        for reward_data_step in reward_data:
            average_rewards.append(tf.reduce_mean(reward_data_step).numpy())
        return average_rewards

    def get_average_final_step_reward_per_epoch(run_data):
        reward_data = run_data['raw_rewards']
        last_step_rewards = []
        for reward_data_step in reward_data:
            last_step_rewards.append(tf.reduce_mean(reward_data_step[:,-1]).numpy())
        return last_step_rewards

    runs_a_average_reward_data = [get_average_reward_per_epoch(rd) for rd in runs_a_data]
    runs_a_final_step_reward_data = [get_average_final_step_reward_per_epoch(rd) for rd in runs_a_data]
    runs_b_average_reward_data = [get_average_reward_per_epoch(rd) for rd in runs_b_data]
    runs_b_final_step_reward_data = [get_average_final_step_reward_per_epoch(rd) for rd in runs_b_data]

    def runs2pd(runs, reward_name: str, runs_type: str):
        rows = [
            {'epoch': epoch_idx + 1, reward_name: reward}
            for run in runs
            for epoch_idx, reward in enumerate(run)
        ]
        runs_df = pd.DataFrame(rows)
        #add column containing runs type
        runs_df['type'] = runs_type
        return runs_df

    runs_a_average_reward_data_df = runs2pd(runs_a_average_reward_data, 'average_reward', runs_desc_a)
    runs_a_final_step_reward_data_df = runs2pd(runs_a_final_step_reward_data, 'final_step_reward', runs_desc_a)
    runs_b_average_reward_data_df = runs2pd(runs_b_average_reward_data, 'average_reward', runs_desc_b)
    runs_b_final_step_reward_data_df = runs2pd(runs_b_final_step_reward_data, 'final_step_reward', runs_desc_b)
    runs_average_reward_data_df = pd.concat([runs_b_average_reward_data_df, runs_a_average_reward_data_df], ignore_index=True)
    runs_final_step_reward_data_df = pd.concat([runs_b_final_step_reward_data_df, runs_a_final_step_reward_data_df], ignore_index=True)


    #seaborn time
    #two plots above each other, for last step and total average reward

    sns.set_style('darkgrid')

    # create two stacked subplots, share x-axis
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 8))

    # helper to plot safely (handles empty dataframes)
    def safe_lineplot(ax, df, x, y, hue, title, xlabel=None, ylabel=None):
        if df is None or df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            return
        sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax)
        ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

    # top: average reward per epoch
    safe_lineplot(
        axes[0],
        runs_average_reward_data_df,
        x='epoch',
        y='average_reward',
        hue='type',
        title='Average reward per epoch',
        xlabel='',  # keep xlabel for the bottom plot
        ylabel='Average reward'
    )

    # bottom: final-step reward per epoch
    safe_lineplot(
        axes[1],
        runs_final_step_reward_data_df,
        x='epoch',
        y='final_step_reward',
        hue='type',
        title='Final-step reward per epoch',
        xlabel='Epoch',
        ylabel='Final-step reward'
    )

    # Consolidate legends: get handles/labels from bottom ax, then remove individual legends
    handles, labels = axes[1].get_legend_handles_labels()
    for ax in axes:
        leg = ax.get_legend()
        if leg:
            leg.remove()

    if handles:
        fig.legend(handles, labels, loc='upper right', title='type')

    plt.tight_layout(rect=[0, 0, 0.92, 1])  # leave room for the legend on the right
    plt.show()

def extract_data_for_probe(save_dir, epoch, sample_epochs_per_extraction, num_extractions, num_parallel):
    four_digit_step = str(epoch).zfill(4)
    model_path = os.path.join(save_dir, 'checkpoints')
    print(model_path)
    model_path = os.path.join(model_path, f'epoch_{four_digit_step}')
    print(model_path)
    model_path = os.path.join(model_path, 'actor.weights.h5')
    print(model_path)
    assert os.path.exists(model_path), f"Model checkpoint not found at: {model_path}"
    #from savedir load config.json
    args = ppo.load_config(os.path.join(save_dir, 'config.json'))
    agent, critic, _, _ = ppo.initialize_agents(actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                                                                       feature_dim=args.feature_dim, embed_dim=args.embed_dim, lstm_units=args.lstm_units,
                                                                       vocab_size=args.vocab_size, msg_len=args.msg_len, which_agent=args.which_agent)
    #load agent weights
    agent.load_weights(model_path)
    shuffle_buffer_size = 512
    prefetch_buffer_size = 4
    dataset = ppo.generate_ppo_dataset(num_same=args.num_same, num_diff1=args.num_diff1, num_diff2=args.num_diff2,
                                   shuffle_buffer_size=shuffle_buffer_size, prefetch_buffer_size=prefetch_buffer_size,
                                   batch_size=num_parallel, which=args.which_dataset)
    num_rollouts = math.ceil(sample_epochs_per_extraction/num_parallel)
    s_1 = []
    s_2 = []
    for _ in range(num_extractions):
        k_rollouts = ppo.do_k_rollouts(dataset,
                                   agent, critic,
                                   reward_function=helpers.target_match_ratio,
                                   num_steps=args.num_steps, k_rollouts=num_rollouts)
        rollouts = ppo.merge_rollouts(k_rollouts)

        should_be_equal = rollouts['agent_1']['input_messages'][:, 1:, :] == rollouts['agent_2']['output_messages'][:, :19, :]
        print(tf.reduce_all(should_be_equal))
        states_agent_1 = tf.concat([rollouts['agent_1']['prev_state_actor_h'], rollouts['agent_1']['prev_state_actor_c']], axis=-1)
        states_agent_2 = tf.concat([rollouts['agent_2']['prev_state_actor_h'], rollouts['agent_2']['prev_state_actor_c']], axis=-1)
        s_1.append(states_agent_1.numpy())
        s_2.append(states_agent_2.numpy())
    s_1 = np.concatenate(s_1, axis=0)
    s_2 = np.concatenate(s_2, axis=0)
    return s_1, s_2

def adjusted_r2(r2, num_samples_n, num_variables_p):
    assert num_samples_n >= num_variables_p, "Number of samples must be greater than or equal to number of variables for adjusted R^2 calculation."
    adj_r2 = 1 - ((1 - r2) * (num_samples_n - 1) / (num_samples_n - num_variables_p - 1))
    return adj_r2

def linear_regression_probe(states_agent_1, states_agent_2):
    #reshape data
    feat_dim = tf.shape(states_agent_1)[-1].numpy()
    states_agent_1 = np.reshape(states_agent_1, (-1, feat_dim))
    states_agent_2 = np.reshape(states_agent_2, (-1, feat_dim))
    linreg = LinearRegression(n_jobs=-1)
    linreg = linreg.fit(states_agent_1, states_agent_2)
    r_squared = linreg.score(states_agent_1, states_agent_2)

    num_samples = states_agent_1.shape[0]
    num_vars = states_agent_1.shape[1] #ingoring intercept, could have normalized first
    adj_r2 = adjusted_r2(r_squared, num_samples, num_vars)

    print(f'R-squared: {r_squared}, Adjusted R-squared: {adj_r2}')
    return linreg

def lin_reg_probe_timesteps(states_agent_1, states_agent_2, probe):
    #Computes r2 over timesteps
    #sklearn r2    feat_dim = tf.shape(states_agent_1)[-1].numpy()
    feat_dim = tf.shape(states_agent_1)[-1].numpy()
    states_agent_1_flat = np.reshape(states_agent_1, (-1, feat_dim))
    states_agent_2_flat = np.reshape(states_agent_2, (-1, feat_dim))
    r2_scores_sk = probe.score(states_agent_1_flat, states_agent_2_flat)
    print(f'R-squared from sk learn: {r2_scores_sk}')
    pred = probe.predict(states_agent_1_flat)
    #reshape pred to original shape
    pred = np.reshape(pred, states_agent_2.shape)
    var_states_agent_2 = np.square(states_agent_2 - np.mean(states_agent_2, axis=(0,1), keepdims=True))
    var_states_agent_2 = np.mean(var_states_agent_2, axis=(0,1), keepdims=True)
    #make sure the minimum of the variance is smaaaaaaall
    variances =np.sort(var_states_agent_2.flatten())
    print(f'min variance of variance vector {np.min(var_states_agent_2)}')
    #count zeros in error message
    #assert np.min(var_states_agent_2) > 1e-6, f"Variance of states_agent_2 is too small, cannot compute R^2 properly. Multiple entries are close to zero: {np.min(var_states_agent_2)}"
    squared_residuals_agent_2 = np.square(states_agent_2 - pred)
    print(np.mean(squared_residuals_agent_2), 'squared residuals, median is:', np.median(squared_residuals_agent_2))
    r2_per_datapoint = 1. - (squared_residuals_agent_2 / var_states_agent_2)
    print(f'self made r2: {np.mean(r2_per_datapoint)}')
    num_steps = states_agent_1.shape[1]
    r2_per_datapoint = np.mean(r2_per_datapoint, axis=-1)
    r2_all_vs_timestep = np.reshape(r2_per_datapoint, (-1, num_steps))
    print(f'resulting shapes; org:{states_agent_2.shape}, result shape: {r2_all_vs_timestep.shape}')
    return r2_all_vs_timestep

def r2_vs_timestep_to_df(r2_vs_timestep, step_base: int = 1) -> pd.DataFrame:
    """
    Convert r2_vs_timestep (shape [samples_per_step, step]) into a tidy DataFrame
    with exactly two columns: 'step' and 'value'.

    Args:
        r2_vs_timestep: 2-D array-like (numpy array, TF tensor, or list-of-equal-length-lists)
                         shape (n_samples, n_steps)
        step_base: 0 or 1. If 1, steps will be 1..n_steps. If 0, steps will be 0..n_steps-1.

    Returns:
        pd.DataFrame with columns ['step', 'value'] and n_samples * n_steps rows.
    """
    # If TF tensor, convert to numpy
    if isinstance(r2_vs_timestep, tf.Tensor):
        r2_vs_timestep = r2_vs_timestep.numpy()

    arr = np.asarray(r2_vs_timestep)

    # Validate shape
    if arr.ndim != 2:
        raise ValueError(f"r2_vs_timestep must be 2-D. Got shape {arr.shape}")

    n_samples, n_steps = arr.shape

    # Build wide DataFrame with string column names '0','1',...
    cols = [str(i) for i in range(n_steps)]
    df_wide = pd.DataFrame(arr, columns=cols)

    # Melt into long/tidy format; no id_vars so every cell becomes a row
    df_long = df_wide.melt(var_name='step', value_name='r2')

    # Convert step from string to integer and apply base offset if requested
    df_long['step'] = df_long['step'].astype(int) + (step_base if step_base in (0,1) else 1)

    # Keep only 'step' and 'value' columns (in that order)
    df_result = df_long[['step', 'r2']].reset_index(drop=True)

    return df_result

def plot_probe_step_results(r2_vs_timestep):
    """
    Args: r2_vs_timestep: np array of shape [samples_per_step, step]
    """
    #process to df
    print('making df')
    df = r2_vs_timestep_to_df(r2_vs_timestep, step_base=0.)
    print('plotting...')
    sns.lineplot(data=df, x='step', y='r2')
    plt.show()



def test_plot_relative():
    runs_a = [
        'runs/2026-03-23_01-25-54',
        'runs/2026-03-23_07-05-30'
    ]
    runs_desc_a = 'separate'
    runs_b = [
        'runs/2026-03-22_02-17-08',
        'runs/2026-03-22_07-54-43',
        'runs/2026-03-22_13-31-34'
    ]
    runs_desc_b = 'shared'
    plot_relative(runs_a,runs_desc_a,runs_b,runs_desc_b)

def test_probe():
    dir = 'runs/2026-03-23_01-25-54'
    step = 1499
    #train data probe
    d_1, d_2 = extract_data_for_probe(dir, step, sample_epochs_per_extraction=2000, num_extractions=2, num_parallel=256)
    probe = linear_regression_probe(d_1, d_2)
    #test data probe
    d_1, d_2 = extract_data_for_probe(dir, step, sample_epochs_per_extraction=2000, num_extractions=2, num_parallel=256)
    r2_timesteps = lin_reg_probe_timesteps(d_1, d_2, probe)
    print('plot now')
    plot_probe_step_results(r2_timesteps)
def main():
    #test_plot_relative()
    test_probe()

if __name__ == '__main__':
    main()
