import tensorflow as tf
import tensorflow_probability as tfp
import os 
import agents
import data 
import helpers


def generate_ppo_dataset(num_same, num_diff1, num_diff2, shuffle_buffer_size, prefetch_buffer_size, batch_size, which='TRAIN'):
    """generates a dataset that contains tuples of features, target pair per agent

    Args:
        num_same (int): Number of objects that are shared between agents
        num_diff1 (int): Number of objects that are only perceived by agent 1
        num_diff2 (int): Number of objects that are only perceived by agent 2
        shuffle_buffer_size (int): Buffer size for shuffling a tf.dataset
        prefetch_buffer_size (int): Buffer size for prefetching elements of a tf.dataset
        batch_size (int): Size of the batch produced
        which (str): String that details whether the train, test, or validation portion of the input dataset is used

    Returns:
        A tensorflow dataset that contains batched version of the input dataset consisting of 
        ((image_features_agent1, target_vector_agent1),(image_features_agent2, target_vector_agent2)) elements
    """

    if which == 'TRAIN':
        path = os.path.join(os.getcwd(), "saved_data/train")
    elif which == 'TEST':
        path = os.path.join(os.getcwd(), "saved_data/test")
    elif which == 'VAL':
        path = os.path.join(os.getcwd(), "saved_data/val")

    dataset = tf.data.Dataset.load(path)
    game_input_ds = data.create_game_instances_dataset(dataset, num_same, num_diff1, num_diff2, shuffle_buffer_size)

    game_input_ds = game_input_ds.shuffle(shuffle_buffer_size)
    game_input_ds = game_input_ds.batch(batch_size)
    game_input_ds = game_input_ds.prefetch(prefetch_buffer_size) 

    return game_input_ds


def rollout_step(agent, critic, features, targets, input_message, reward_function):
    """a function that produces a single rollout step 

    Args:
        agent (tf.keras.Model): A reinforcement learning agent from agents.py
        critic (tf.keras.Model): A reinforcement learning critic from agents.py
        features (tensor): A set of features describing images seen by the agent; shape [batch_size, num_imgs, feature_dim]
        targets (tensor): A binary tensor that details whether a feature tensor is a target (=1) or not (=0); shape [batch_size, num_img]
        input_message (tensor): A tensor of discrete symbols that was sent by the other agent (or an initial message); shape [batch_size, message_length]
        num_steps (int): The number of steps the agents takes before the game is terminated
        reward_function (function): some reward function based on correct target prediction 
            - takes predictions and targets as inputs

    Returns:
        A list of dicts containing values collected during each rollout step
            - features: the same features from the input
            - output_messages: message sampled from 
            - preds: the agents prediction on what images are targets
            - img_logps:log probabilities based on the probability distribution over the images
            - msg_logps: log probabilities based on the probability distribution over the message
            - joint_logps: combined log probability of img_logps and msg_logps
            - rewards: Rewards returned by the reward function based on the correct target prediction
            - values: Critics predicted value based on image and message inputs
    """
    probs_img, probs_msg = agent(features, input_message)
    vals = critic(features, input_message)

    img_dist = tfp.distributions.Categorical(probs=probs_img)
    output_messages = img_dist.sample() 
    img_logps =img_dist.log_prob(output_messages)

    msg_dist = tfp.distributions.Bernoulli(probs=probs_msg) # to be adjusted with longer message length
    preds = msg_dist.sample()
    msg_logps = msg_dist.log_prob(preds)
    
    preds = tf.cast(preds, dtype=tf.float32)

    rewards = reward_function(preds, targets)
    joint_logps = tf.reduce_sum(img_logps, axis=-1) + tf.reduce_sum(msg_logps, axis=-1)
    
    return {
         "features": features,                  # [batch_size, num_imgs, feature_dim]
         "input_messages": input_message,       # [batch_size,] --> for now, only one symbol is returned as message, will adjust later
         "output_messages": output_messages,    # [batch_size,] --> for now, only one symbol is returned as message, will adjust later
         "preds": preds,                        # [batch_size, num_imgs]
         "img_logps": img_logps,                # [batch_size,]
         "msg_logps": msg_logps,                # [batch_size, num_imgs]
         "joint_logps": joint_logps,            # [batch_size,]
         "rewards": rewards,                    # [batch_size,]
         "values": vals,                        # [batch_size,]
       }

def do_n_rollout_steps(agent_1, critic_1, features_a1, targets_a1, 
                        agent_2, critic_2, features_a2, targets_a2,
                        reward_function, num_steps):
    
    """collects num_steps many rollout steps for both agents and merges all into a single dictionary

    Args:
        agent_1, agent_2 (tf.keras.Model): Reinforcement learning agents from agents.py
        critic_1, critic_2 (tf.keras.Model): Reinforcement learning critics from agents.py
        features_a1, features_a2 (tensor): Sets of features describing images seen by the agents respectively [batch_size, num_img, feature_dim]
        targets_a1, targets_a2 (tensor): Binary tensors that details whether a feature tensor is a target (=1) or not (=0), for each agent respectively; shape [batch_size, num_img]
        num_steps (int): The number of steps until the rollout collection is terminated
        
    Return:
        A dictionary with fused timesteps of num_step many dictionaries that contain the values of the rollout steps for each agent 
    """
    batch_size = tf.shape(features_a1)[0]

    two_agents_rollout = {
        "agent_1": {key: [] for key in ["features", "input_messages", "output_messages", "preds", "img_logps", "msg_logps", "joint_logps", "rewards", "values"]},
        "agent_2": {key: [] for key in ["features", "input_messages", "output_messages", "preds", "img_logps", "msg_logps", "joint_logps", "rewards", "values"]},
    }

    for step in range(num_steps):
        
        if step == 0:
            input_message_a1 = tf.zeros([batch_size], dtype=tf.int32)   # To do: adjust to longer message length
            input_message_a2 = tf.zeros([batch_size], dtype=tf.int32)   # To do: adjust to longer message length
        else:
            input_message_a1 = two_agents_rollout["agent_2"]["output_messages"][step-1]
            input_message_a2 = two_agents_rollout["agent_1"]["output_messages"][step-1]


        rollout_step_a1 = rollout_step(agent_1, critic_1, features_a1, targets_a1, input_message_a1, reward_function)
        rollout_step_a2 = rollout_step(agent_2, critic_2, features_a2, targets_a2, input_message_a2, reward_function)

        for k, v in rollout_step_a1.items():
            two_agents_rollout["agent_1"][k].append(v)
        for k, v in rollout_step_a2.items():
            two_agents_rollout["agent_2"][k].append(v)
        
    for agent in ["agent_1", "agent_2"]:
        for k in two_agents_rollout[agent]:
            two_agents_rollout[agent][k] = tf.stack(two_agents_rollout[agent][k], axis=0)

    return two_agents_rollout

    
def do_k_rollouts(feature_dataset, agent_1, critic_1, agent_2, critic_2, reward_function, num_steps, num_envs):
    """A function that carries out k many separate rollouts
    Args:
        feature_dataset(tf.dataset): consisting of ((image_features_agent1, target_vector_agent1),(image_features_agent2, target_vector_agent2)) elements
        agent_1, agent_2 (tf.keras.Model): Reinforcement learning agents from agents.py
        critic_1, critic_2 (tf.keras.Model): Reinforcement learning critics from agents.py
        num_steps (int): The number of steps until the rollout collection is terminated
        k (int): number of environments (games) from which rollouts are collected
    
    Return:
        A list of k rollouts with each element being a dict that contains all rollout information for both agents over num_step timesteps
    """
    k_rollouts = []
    for (features_a1, targets_a1), (features_a2, targets_a2) in feature_dataset.take(num_envs):
        rollout_steps = do_n_rollout_steps(agent_1, critic_1, features_a1, targets_a1,
                                                agent_2, critic_2, features_a2, targets_a2,
                                                reward_function, num_steps)
        k_rollouts.append(rollout_steps)

    return k_rollouts

        

def merge_rollouts(k_rollouts):
    """merges all k*m rollouts along the rollout parameters
    
    Args:
        k_rollouts (list): List of k many rollouts with m minibatches and n rollout steps each
    Return:
        Dictionary that contains the merged rollout information
    """
    merged_rollouts = {
        "agent_1": {},
        "agent_2": {},
    }

    for a in ["agent_1", "agent_2"]:
        for key in k_rollouts[0]["agent_1"].keys():
            merged = tf.concat([r[a][key] for r in k_rollouts], axis=1)
            # transposing in such a way that batch_size is first and num_steps second
            perm = tf.concat([[1, 0], tf.range(2, tf.rank(merged))], axis=0)
            merged = tf.transpose(merged, perm)
            

            merged_rollouts[a][key] = merged


    return merged_rollouts

def prepare_ppo_information(merged, critic_1, critic_2):
    """Calculates and adds addvantage and return information to the merged rollout dictionary
    Args:
        merged (dict): contains the merged rollout information
        critic_1, critc_2 (tf.keras.Model): Reinforcement learning critics from agents.py; here for advantages bootstrapping
        
    Return:
        The dict with the rollout information has been expanded by returns and advantages
    """
    # accessing the last feature sets and messsages for bootstrapping
    last_features_a1 = merged["agent_1"]["features"][:, -1]
    last_features_a2 = merged["agent_2"]["features"][:, -1]
    last_msg_a1 = merged["agent_2"]["output_messages"][:, -1] 
    last_msg_a2 = merged["agent_1"]["output_messages"][:, -1] 
    # creating the bootstrapping values 
    last_val_a1 = critic_1(last_features_a1, last_msg_a1)
    last_val_a2 = critic_2(last_features_a2, last_msg_a2)

    all_vals_a1 = tf.concat([merged["agent_1"]["values"], last_val_a1[:, None]], axis=1)
    all_vals_a2 = tf.concat([merged["agent_2"]["values"], last_val_a2[:, None]], axis=1)
    
    returns_a1 , advantages_a1 = helpers.compute_gae(merged["agent_1"]["rewards"],all_vals_a1)
    returns_a2 , advantages_a2 = helpers.compute_gae(merged["agent_2"]["rewards"], all_vals_a2)
    
    merged["agent_1"]["returns"] = returns_a1           # shape [batch_size,]
    merged["agent_1"]["advantages"] = advantages_a1     # shape [batch_size,]
    merged["agent_2"]["returns"] = returns_a2           # shape [batch_size,]
    merged["agent_2"]["advantages"] = advantages_a2     # shape [batch_size,]


    return merged

def rollouts_to_dataset(merged, buffer_size, batch_size):    
    """
    Args:
        merged (dict): A dict that contains all PPO information from the rollout relevant for the training step
        buffer_size (int): Buffer size for shuffling a tf.dataset
        batch_size (int): Size of the batch produced
    
    Return:
        A tf.dataset that contains all PPO information from the rollout relevant for the training step"""
    def flatten(x):
        """manual flattening for easier dataset slicing"""
        x_shape = tf.shape(x)
        return tf.reshape(x, tf.concat([[x_shape[0] * x_shape[1]], x_shape[2:]], axis=0))

    a1_flat = {k: flatten(v) for k, v in merged["agent_1"].items()}
    a2_flat = {k: flatten(v) for k, v in merged["agent_2"].items()}

    dataset = tf.data.Dataset.from_tensor_slices({
        "agent_1": a1_flat,                         # each key now has (batch_size*num_steps) as first dimension
        "agent_2": a2_flat,
    })
    dataset = dataset.shuffle(buffer_size).batch(batch_size)

    return dataset

@tf.function
def train_step(rollout_data, 
                agent, critic, optimizer_agent, optimizer_critic, 
                clip_epsilon=0.2, entropy_coef=0.01, which_agent=""):
    """train step for PPO training between two agents in a bidiriectional multistep referential game
     Args:
        rollout_data (tf.dataset): A tf.dataset that contains all PPO information from the rollout relevant for the training step
        agent (tf.keras.Model):  A reinforcement learning agent from agents.py
        critic (tf.keras.Model): A reinforcement learning critic from agents.py
        optimizer_agent, optimizer_critic (tf.keras.optimizers): Optimizers that update the parameters of the agent (actor) and critic networks
        clip_epsilon (float): a hyperparameter to constrain the policy update
        entropy_coef (float): hyperparameter to constrain entropy bonus
        which_agent (str): A string that denotes which agent's rollout is used (to be able to get the message from the other agent)

    Return:
        actor_loss_a1, critic_loss_a1, entropy_a1, actor_loss_a2, critic_loss_a2, entropy_a2
    """

    with tf.GradientTape(persistent=True) as tape:
        probs_img, probs_msg = agent(rollout_data[which_agent]["features"], rollout_data[which_agent]["input_messages"])
        vals = critic(rollout_data[which_agent]["features"], rollout_data[which_agent]["input_messages"])

        img_dist = tfp.distributions.Categorical(probs=probs_img)
        output_messages = img_dist.sample() 
        img_logps =img_dist.log_prob(output_messages)

        msg_dist = tfp.distributions.Bernoulli(probs=probs_msg) # to be adjusted with longer message length
        preds = msg_dist.sample()
        msg_logps = msg_dist.log_prob(preds)

        preds = tf.cast(preds, dtype=tf.float32)
    
        joint_logps = tf.reduce_sum(img_logps, axis=-1) + tf.reduce_sum(msg_logps, axis=-1)

        ratios = tf.exp(joint_logps - rollout_data[which_agent]["joint_logps"])
        entropy = tf.reduce_mean(img_dist.entropy()) + tf.reduce_sum(msg_logps, axis=-1)

        unclipped = ratios * rollout_data[which_agent]["advantages"]
        clipped = tf.clip_by_value(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * rollout_data[which_agent]["advantages"]

        actor_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped)) - entropy_coef * entropy
        critic_loss = tf.reduce_mean(tf.square(rollout_data[which_agent]["returns"] - vals))

    # Compute gradients
    agent_grads = tape.gradient(actor_loss, agent.trainable_variables)
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)

    optimizer_agent.apply_gradients(zip(agent_grads, agent.trainable_variables))
    optimizer_critic.apply_gradients(zip(critic_grads, critic.trainable_variables))
    
    del tape

    return actor_loss, critic_loss


agent_1 = agents.AgentDummy()
agent_2 = agents.AgentDummy()

critic_1 = agents.AgentDummyCritic()
critic_2 = agents.AgentDummyCritic()

optimizer_agent_1 = tf.keras.optimizers.Adam(1e-2) #1e-3
optimizer_agent_2 = tf.keras.optimizers.Adam(1e-2) #1e-3
optimizer_crit_1 = tf.keras.optimizers.Adam(1e-2) #1e-3
optimizer_crit_2 = tf.keras.optimizers.Adam(1e-2) #1e-3


def train(agent_1, agent_2, critic_1, critic_2, 
          optimizer_agent_1,optimizer_agent_2,optimizer_crit_1,optimizer_crit_2, 
          num_same, num_diff1, num_diff2, shuffle_buffer_size, prefetch_buffer_size, batch_size, which_dataset,
          num_steps, num_envs):
    
    dataset = generate_ppo_dataset(num_same=num_same, num_diff1=num_diff1, num_diff2=num_diff2, 
                                    shuffle_buffer_size=shuffle_buffer_size, prefetch_buffer_size=prefetch_buffer_size, batch_size=batch_size, which=which_dataset)
    k_rollouts = do_k_rollouts(dataset, 
                             agent_1, critic_1, 
                             agent_2, critic_2, 
                             reward_function=helpers.target_match_ratio, 
                             num_steps=num_steps, num_envs=num_envs)
    merged_rollouts = merge_rollouts(k_rollouts)
    merged_rollouts = prepare_ppo_information(merged_rollouts, critic_1, critic_2)
    rollout_dataset = rollouts_to_dataset(merged_rollouts, buffer_size=shuffle_buffer_size, batch_size=batch_size)

    # explicitly initializing optimizer slots to avoid singleton variable error (due to train_step call with second model)
    helpers.initialize_optimizer_slots(optimizer_agent_1, agent_1)
    helpers.initialize_optimizer_slots(optimizer_crit_1, critic_1)
    helpers.initialize_optimizer_slots(optimizer_agent_2, agent_2)
    helpers.initialize_optimizer_slots(optimizer_crit_2, critic_2)


    # Convert dataset to iterator (can be repeated and prefetched)
    iterator = iter(rollout_dataset)
    num_batches = tf.data.experimental.cardinality(rollout_dataset)
    
    # TensorArrays to store losses
    actor_losses_1 = tf.TensorArray(tf.float32, size=num_batches, element_shape=None)
    critic_losses_1 = tf.TensorArray(tf.float32, size=num_batches, element_shape=None)
    actor_losses_2 = tf.TensorArray(tf.float32, size=num_batches, element_shape=None)
    critic_losses_2 = tf.TensorArray(tf.float32, size=num_batches, element_shape=None)
    
    for i in tf.range(num_batches):
        batch = next(iterator)
        
        # Train agent 1
        actor_loss_1, critic_loss_1 = train_step(batch, agent_1, critic_1,
                                                 optimizer_agent_1, optimizer_crit_1,
                                                 clip_epsilon=0.2, entropy_coef=0.01,
                                                 which_agent="agent_1")
        # Train agent 2
        actor_loss_2, critic_loss_2 = train_step(batch, agent_2, critic_2,
                                                 optimizer_agent_2, optimizer_crit_2,
                                                 clip_epsilon=0.2, entropy_coef=0.01,
                                                 which_agent="agent_2")
        
        # Write into TensorArrays
        actor_losses_1 = actor_losses_1.write(i, actor_loss_1)
        critic_losses_1 = critic_losses_1.write(i, critic_loss_1)
        actor_losses_2 = actor_losses_2.write(i, actor_loss_2)
        critic_losses_2 = critic_losses_2.write(i, critic_loss_2)
    
    # Stack TensorArrays to tensors
    actor_losses_1 = actor_losses_1.stack()
    critic_losses_1 = critic_losses_1.stack()
    actor_losses_2 = actor_losses_2.stack()
    critic_losses_2 = critic_losses_2.stack()
    
    return actor_losses_1, critic_losses_1, actor_losses_2, critic_losses_2
 
actor_losses_1, critic_losses_1, actor_losses_2, critic_losses_2 = train(agent_1, agent_2, critic_1, critic_2, 
          optimizer_agent_1,optimizer_agent_2,optimizer_crit_1,optimizer_crit_2, 
          num_same=5, num_diff1=3, num_diff2=3, shuffle_buffer_size=1000, prefetch_buffer_size=1000, batch_size=10, which_dataset='TRAIN',
          num_steps=10, num_envs=4)

# print(actor_losses_1)