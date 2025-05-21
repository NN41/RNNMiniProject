# %%

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using {device} device\n")

# %% Load data stored on disk

DATA_NAME = "baseprice" # or funding
DATA_FILE_NAME = "deribit_btc_perpetual_tradingview_ohlc_res_60_20230101_to_20250517.csv"
DATA_FILE_PATH = os.path.join("C:\\Users\\nickn\\AlgoTrading\\DataScraper\\data", DATA_FILE_NAME)
TS_COLUMN_NAME = "datetime_utc"

print(f"Loading {DATA_NAME} data from: {DATA_FILE_PATH}")

try:
    df = pd.read_csv(DATA_FILE_PATH, parse_dates=[TS_COLUMN_NAME])
    print("Data loaded successfully.")
    print("\n--- DataFrame Info ---")
    df.info()
    print("\n--- DataFrame Head ---")
    print(df.head(3))
    print("\n--- DataFrame Tail ---")
    print(df.tail(3))
except FileNotFoundError:
    print(f"ERROR: Data file not found at {DATA_FILE_PATH}")
    print("Please ensure the DataScraper script has run and the path/filename is correct.")
    exit()
except Exception as e:
    print(f"An error occurred while loading or processing the data: {e}")
    exit()

df.set_index(TS_COLUMN_NAME, inplace=True)
df.sort_index(inplace=True)
df_raw = df[['open','high','low','close']]

# df = df_raw
# df_o_transformed = df[['open']].div(df['open'].shift(), axis=0) - 1
# df_hlc_transformed = df[['high','low','close']].div(df['open'], axis=0) - 1
# df_X = pd.merge(df_o_transformed, df_hlc_transformed, left_index=True, right_index=True).dropna()
# df_y = ((df[['open']].shift(-1) - df[['open']]).dropna() >= 0).astype(int).rename(columns={'open':'target'})
# df_Xy = pd.merge(df_X, df_y, left_index=True, right_index=True)

# df_Xy


# %%

df_24h = df_raw[['open']][df_raw.index.hour == 8]

df_input = (df_24h / df_24h.shift() - 1) # percentage change of the open price
df_input = df_input[1:-1] # handle nans

df_target = ((-df_24h.diff(-1)) >= 0).astype(int) # 1 if up-move, 0 of down-move
df_target = df_target[1:-1] # handle nans

# %%

window_size = 3
n_samples = df_input.shape[0]

frac_train = 0.8
frac_test = 1 - frac_train

train_start_idx = int(window_size - 1)
train_end_idx = int(n_samples * frac_train)
test_start_idx = int(train_end_idx + window_size)
test_end_idx = n_samples - 1

print(f"")
print(f"number of samples = {n_samples}")
print(f"window size = {window_size}")
print(f"train: (start, end) = {train_start_idx, train_end_idx} | test: (start, end) = {test_start_idx, test_end_idx}")

# %%

X_train = []
for idx in range(train_start_idx, train_end_idx+1):
    sequence_start_idx = idx - window_size + 1
    sequence_end_idx = idx
    input_sequence = df_input.iloc[sequence_start_idx:sequence_end_idx+1].to_numpy()
    X_train.append(input_sequence)
X_train = torch.tensor(np.array(X_train)).to(device)
y_train = torch.tensor(df_target[train_start_idx:train_end_idx+1].to_numpy()).to(device)

X_test = []
for idx in range(test_start_idx, test_end_idx+1):
    sequence_start_idx = idx - window_size + 1
    sequence_end_idx = idx
    input_sequence = df_input.iloc[sequence_start_idx:sequence_end_idx+1].to_numpy()
    X_test.append(input_sequence)
X_test = torch.tensor(np.array(X_test)).to(device)
y_test = torch.tensor(df_target[test_start_idx:test_end_idx+1].to_numpy()).to(device)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
