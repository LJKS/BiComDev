import csv
import pandas as pd
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import helpers

##### ONLY WORKS FOR 1 TARGET, 1 DISTRACTOR PAIRS. USED FOR DEBUGGING #####



run_dir="/home/vanfra/recovered files/bidirectional_signallling_game/runs/2026-02-17_23-40-42"

epoch_data_path = os.path.join(run_dir, "training_epochs.csv")
step_data_path = os.path.join(run_dir, "training_steps.csv")

data_epoch = pd.read_csv(epoch_data_path, index_col="epoch")

print(np.max(data_epoch["train_reward"]))

n = 5
top_n = np.partition(data_epoch["train_reward"], -n)[-n:]
print(top_n)



path = os.path.join(run_dir, "raw_data.pkl")
with open(path, "rb") as file:
    data = pickle.load(file)


def analyze_predictions(preds, targets):
    """
    preds: [batch, steps, 2]  (binary)
    targets: [batch, steps]   (0 or 1)
    """

    preds = preds.reshape(-1, 2)
    targets = targets.reshape(-1)

    mask_10 = np.all(preds == [1, 0], axis=1)
    mask_01 = np.all(preds == [0, 1], axis=1)
    mask_11 = np.all(preds == [1, 1], axis=1)
    mask_00 = np.all(preds == [0, 0], axis=1)

    dist = {
        "[1,0]": mask_10.mean(),
        "[0,1]": mask_01.mean(),
        "[1,1]": mask_11.mean(),
        "[0,0]": mask_00.mean(),
    }

    correct_10 = (targets[mask_10] == 0).mean() if mask_10.any() else 0
    correct_01 = (targets[mask_01] == 1).mean() if mask_01.any() else 0

    return dist, correct_10, correct_01



def tensor_str_to_float(x):
    if isinstance(x, str):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x)
        if match:
            return float(match.group())
        else:
            raise ValueError(f"Could not parse float from: {x}")
    return float(x)


training_rewards = data_epoch["train_reward"].apply(tensor_str_to_float)
training_accuracies = data_epoch["train_accuracies"].apply(tensor_str_to_float)
actor_losses = data_epoch["mean_actor_loss"].apply(tensor_str_to_float)
critic_losses = data_epoch["mean_critic_loss"].apply(tensor_str_to_float)
eval_rewards=None



preds_per_epoch = data["preds"]

epoch_rows = []

for epoch, preds in enumerate(preds_per_epoch):
    preds = np.array(preds)             
    flat = preds.reshape(-1, 2)           
    total = len(flat)

    counts = Counter(map(tuple, flat))

    full_reward = (
        counts.get((1.0, 0.0), 0) +
        counts.get((0.0, 1.0), 0)
    )

    half_reward = (
        counts.get((1.0, 1.0), 0) +
        counts.get((0.0, 0.0), 0)
    )

    epoch_rows.append({
        "epoch": epoch,
        "full_reward_frac": full_reward / total,
        "ambiguous_frac": half_reward / total,
        "pred_[1,0]": counts.get((1.0, 0.0), 0) / total,
        "pred_[0,1]": counts.get((0.0, 1.0), 0) / total,
        "pred_[1,1]": counts.get((1.0, 1.0), 0) / total,
        "pred_[0,0]": counts.get((0.0, 0.0), 0) / total,
    })


df = pd.DataFrame(epoch_rows)


plt.figure(figsize=(10, 6))

plt.plot(df["epoch"], df["full_reward_frac"], label="Full reward (1-hot)", linewidth=2)
plt.plot(df["epoch"], df["ambiguous_frac"], label="Half reward (ambiguous)", linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Fraction of predictions")
plt.title("Prediction mode usage over training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# plotting different prediction combinations einzeln
plt.figure(figsize=(10, 6))

plt.plot(df["epoch"], df["pred_[1,0]"], label="[1, 0]")
plt.plot(df["epoch"], df["pred_[0,1]"], label="[0, 1]")
plt.plot(df["epoch"], df["pred_[1,1]"], label="[1, 1]")
plt.plot(df["epoch"], df["pred_[0,0]"], label="[0, 0]")

plt.xlabel("Epoch")
plt.ylabel("Fraction of predictions")
plt.title("Exact prediction pattern frequencies")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


