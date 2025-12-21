import csv
import pandas as pd
import os
import pickle

run_dir="/home/vanfra/recovered files/bidirectional_signallling_game/runs/2025-12-20_23-18-51"

epoch_data_path = os.path.join(run_dir, "training_epochs.csv")
step_data_path = os.path.join(run_dir, "training_epochs.csv")


path = os.path.join(run_dir, "raw_data.pkl")
with open(path, "rb") as file:
    data = pickle.load(file)

print(max(data["raw_rewards"]))