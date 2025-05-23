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
df_raw = df[['open','high','low','close']].astype(np.float32)

# %%

# Note that the row with OHLC data at time t corresponds to the OHLC over the interval [t,t+1]. This means that
# if we observe the OHLC data corresponding to time t, we must actually be at time t+1. 
# 
# We prepare the input data at time t by computing percentage returns of H, L, C with respect to O of the same candle
# and the percentage returns of O with respect to previous O. We also shift it forward by 1, such that the row at time t
# represents the our knowledge at time t.
#
# Our target is predicting if C goes above or below O of within the same interval.
df = df_raw

df_hlc_transformed = (df[['high','low','close']].div(df['open'], axis=0) - 1)
df_o_transformed = (df[['open']].div(df['open'].shift(), axis=0) - 1)
df_input = pd.concat([df_o_transformed, df_hlc_transformed], axis=1).shift() # has NaNs at rows indices 0 and 1
df_target = ((df['close'] - df['open']) >= 0)

df_input = df_input.iloc[2:].astype(np.float32)
df_target = df_target.iloc[2:].astype(int)

df_input.head(), df_target.head()

# %%

window_size = 30 * 24
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
print(f"train samples = {train_end_idx-train_start_idx+1} | test samples = {test_end_idx-test_start_idx+1}")


# %%

X_train = []
for idx in range(train_start_idx, train_end_idx+1):
    sequence_start_idx = idx - window_size + 1
    sequence_end_idx = idx
    input_sequence = df_input.iloc[sequence_start_idx:sequence_end_idx+1].to_numpy()
    X_train.append(input_sequence)
X_train = torch.tensor(np.array(X_train), dtype=torch.float32, device=device)
y_train = torch.tensor(df_target[train_start_idx:train_end_idx+1].to_numpy(), device=device)

X_test = []
for idx in range(test_start_idx, test_end_idx+1):
    sequence_start_idx = idx - window_size + 1
    sequence_end_idx = idx
    input_sequence = df_input.iloc[sequence_start_idx:sequence_end_idx+1].to_numpy()
    X_test.append(input_sequence)
X_test = torch.tensor(np.array(X_test), dtype=torch.float32, device=device)
y_test = torch.tensor(df_target[test_start_idx:test_end_idx+1].to_numpy(), device=device)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              generator=torch.Generator().manual_seed(0))
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"window size = {window_size} | train examples = {len(train_dataset)} | test examples = {len(test_dataset)}")
# %%

import torch.nn as nn
import torch.nn.functional as F

n_features = X_train.shape[-1]
n_hidden = 16
n_classes = 2

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True) # accepts (batch_size, seq_len, n_features)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # h_seq contains all hidden outputs over the entire sequence, h_t is only the last hidden output
        # we only need h_t to make a prediction at time t
        h_seq, h_t = self.rnn(x)
        logits_t = self.h2o(h_t.squeeze()) # h_t is of shape (1, batch_size, n_hidden), so we can discard the first dim
        return logits_t # of shape (batch_size, n_classes)

rnn = RNN(n_features, n_hidden, n_classes).to(device)
# print(rnn)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # h_seq has shape (batch_size, seq_len, hidden_size) because batch_first=True
        # h_n and c_n have shape (1, batch_size, hidden_size)
        # logits_t has shape (batch_size, output_size)
        h_seq, (h_t, c_t) = self.lstm(x)
        logits_t = self.h2o(h_t.squeeze())
        return logits_t

lstm = LSTM(n_features, n_hidden, n_classes).to(device)


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_seq, h_t = self.gru(x)
        logits_t = self.h2o(h_t.squeeze())
        return logits_t

# %%

# model = rnn
model = lstm

model.eval()
with torch.no_grad(): 
    input_sequence, target = train_dataset[200:220]
    logits = model(input_sequence)
    probabilities = nn.Softmax(dim=-1)(logits)
    classes = probabilities.argmax(dim=-1)

logits, probabilities, classes, target

# %%

def test(model, criterion):

    loss_total = 0
    n_correct = 0
    n_samples = 0

    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)   
            logits = model(X)
            n_samples += len(y)
            loss_total += criterion(logits, y).item() * len(y)
            n_correct += (logits.argmax(dim=-1) == y).sum().item()

    avg_loss = loss_total / n_samples
    accuracy = n_correct / n_samples
    return avg_loss, accuracy

def train(model, criterion, optimizer, n_updates=0):

    n_batches = len(train_dataloader)
    # n_updates = 3
    update_batches = np.linspace(0, n_batches-1, n_updates).astype(int)
    total_loss_between_updates = 0
    total_loss = 0
    n_samples_between_updates = 0
    n_samples = 0

    # if n_updates > 0:
    #     print(f"[batch] / {n_batches} | [avg train loss between updates]")
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # forward pass
        logits = model(X)
        loss = criterion(logits, y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()

        n_samples_between_updates += len(y)
        n_samples += len(y)
        total_loss += loss.item() * len(y)
        total_loss_between_updates += loss.item() * len(y)
        if batch in update_batches:
            avg_loss_between_updates = total_loss_between_updates / n_samples_between_updates
            print(f"\t{batch+1} / {n_batches} | avg train loss {avg_loss_between_updates:.5f}")
            n_samples_between_updates = 0
            total_loss_between_updates = 0
    
    avg_train_loss = total_loss / n_samples
    return avg_train_loss

def get_tensor_metrics(tensor):

    metrics_dict = {
        "l2": torch.norm(tensor, p=2).item(),
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "abs_min": tensor.abs().min().item(),
        "abs_max": tensor.abs().max().item(),
    }

    for quantile in [0.5, 0.1, 0.01, 0.001]:
        key = f"abs_q_{quantile}"
        metrics_dict[key] = torch.quantile(tensor.abs(), quantile).item()
        
    return metrics_dict

def get_param_metrics(param):

    param_data_metrics = get_tensor_metrics(param.data)
    param_grad_metrics = get_tensor_metrics(torch.zeros(param.data.shape))
    if not param.grad is None:
        param_grad_metrics = get_tensor_metrics(param.grad)

    param_metrics = {
        "data": param_data_metrics,
        "grad": param_grad_metrics,
    }

    return param_metrics

def get_all_param_metrics(model):
    dic = {}
    for name, param in model.named_parameters():
        dic[name] = get_param_metrics(param)
    return dic

def flatten_nested_dict(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_nested_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


# %%

# model = RNN(input_size=X_train.shape[-1], hidden_size=32, output_size=2).to(device)
model = LSTM(input_size=X_train.shape[-1], hidden_size=4, output_size=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=0.001, 
    # weight_decay=0.00001 # 0.01 is too high and the model doesn't improve, 0.0001 is pretty good
)
print(model)

# %%

EPOCHS = 1000

test_loss, test_acc = test(model, criterion)
print(f"avg test loss = {test_loss:.5f} | accuracy = {test_acc * 100:.2f}%")

logs = []
logs_flattened = []
initial_log = {
    "epoch": 0,
    "test_loss": test_loss,
    "test_acc": test_acc,
    "train_loss": None,
    "": get_all_param_metrics(model),
}
logs.append(initial_log)
logs_flattened.append(flatten_nested_dict(initial_log, separator='/'))

for epoch in range(EPOCHS):
    print(f"\nepoch {epoch+1} / {EPOCHS}")

    train_loss = train(model, criterion, optimizer, n_updates=5)

    test_loss, test_acc = test(model, criterion)
    print(f"avg test loss = {test_loss:.5f} | accuracy = {test_acc * 100:.2f}%")

    log = {
        "epoch": epoch+1,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "train_loss": train_loss,
        "": get_all_param_metrics(model),
    }
    logs.append(log)
    logs_flattened.append(flatten_nested_dict(log, separator='/'))


# %%
df_logs = pd.DataFrame(logs_flattened).set_index('epoch')

rows = 4
cols = 2
figsize_mult = 8
fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_mult, rows * figsize_mult))

ax = axes[0,0]
cols = ['train_loss','test_loss']
ax.plot(df_logs[cols], label=cols)
ax.legend()
ax.grid()
# ax.set_ylim((0.687,0.705))

ax = axes[0,1]
cols = ['test_acc']
ax.plot(df_logs[cols], label=cols)
ax.legend()
ax.grid()

ax = axes[1,0]
cols = [col for col in df_logs.columns if 'grad/l2' in col]
ax.plot(df_logs[cols], label=cols)
ax.legend()
ax.grid()

ax = axes[1,1]
cols = [col for col in df_logs.columns if 'grad/abs_max' in col]
ax.plot(df_logs[cols], label=cols)
ax.legend()
ax.grid()

ax = axes[2,0]
cols = [col for col in df_logs.columns if 'data/l2' in col]
ax.plot(df_logs[cols], label=cols)
ax.legend()
ax.grid()

ax = axes[2,1]
cols = [col for col in df_logs.columns if 'data/abs_max' in col]
ax.plot(df_logs[cols], label=cols)
ax.legend()
ax.grid()

ax = axes[3,0]
cols = [col for col in df_logs.columns if 'grad/abs_q_0.01' in col]
ax.plot(df_logs[cols], label=cols)
ax.legend()
ax.grid()

ax = axes[3,1]
cols = [col for col in df_logs.columns if 'data/abs_q_0.01' in col]
ax.plot(df_logs[cols], label=cols)
ax.legend()
ax.grid()

# %%

# df_logs[]
# [col for col in df_logs.columns if 'grad/l2' in col]
