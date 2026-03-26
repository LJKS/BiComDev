import helpers
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import json
import math
import ppo


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

def extract_data_for_probe(save_dir, epoch, sample_epochs, num_parallel):
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
    shuffle_buffer_size = 512
    prefetch_buffer_size = 4
    dataset = ppo.generate_ppo_dataset(num_same=args.num_same, num_diff1=args.num_diff1, num_diff2=args.num_diff2,
                                   shuffle_buffer_size=shuffle_buffer_size, prefetch_buffer_size=prefetch_buffer_size,
                                   batch_size=num_parallel, which=args.which_dataset)
    num_rollouts = math.ceil(sample_epochs/num_parallel)
    k_rollouts = ppo.do_k_rollouts(dataset,
                               agent, critic,
                               reward_function=helpers.target_match_ratio,
                               num_steps=args.num_steps, k_rollouts=num_rollouts)
    rollouts = ppo.merge_rollouts(k_rollouts)

    should_be_equal = rollouts['agent_1']['input_messages'][:, 1:, :] == rollouts['agent_2']['output_messages'][:, :19, :]
    print(tf.reduce_all(should_be_equal))

    state_zero_h = rollouts['agent_1']['prev_state_actor_h'][:, 0, :]
    state_zero_c = rollouts['agent_2']['prev_state_actor_c'][:, 0, :]
    print('a')
    print('as')

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
    extract_data_for_probe()

def test_probe():
    dir = 'runs/2026-03-23_01-25-54'
    step = 1499
    extract_data_for_probe(dir, step, sample_epochs=1000, num_parallel=128)

def main():
    #test_plot_relative()
    test_probe()

if __name__ == '__main__':
    main()
