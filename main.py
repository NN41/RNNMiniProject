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
        logits = self.h2o(h_t[0]) # h_t is of shape (1, batch_size, n_hidden), so we can discard the first dim
        return logits # of shape (batch_size, n_classes)

rnn = RNN(n_features, n_hidden, n_classes).to(device)
print(rnn)

# %%

# XX, yy = train_dataset[100:120]
# target = yy.flatten()
# logits = rnn(XX)
# probabilities = nn.Softmax(dim=-1)(logits)
# classes = probabilities.argmax(dim=-1)

# logits, probabilities, classes, target
# %%

def test(model, criterion):

    loss_total = 0
    n_correct = 0
    n_samples = 0

    model.eval()

    for batch, (X, y) in enumerate(test_dataloader):
        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            logits = model(X)
            target = y.flatten()
            n_samples += len(target)
            loss_total += criterion(logits, target).item() * len(target)
            n_correct += (logits.argmax(dim=-1) == target).sum().item()

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

    model.train()

    # if n_updates > 0:
    #     print(f"[batch] / {n_batches} | [avg train loss between updates]")
    
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # forward pass
        target = y.flatten()
        logits = model(X)
        loss = criterion(logits, target)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_samples_between_updates += len(target)
        n_samples += len(target)
        total_loss += loss.item() * len(target)
        total_loss_between_updates += loss.item() * len(target)
        if batch in update_batches:
            avg_loss_between_updates = total_loss_between_updates / n_samples_between_updates
            print(f"\t{batch+1} / {n_batches} | avg train loss {avg_loss_between_updates:.5f}")
            n_samples_between_updates = 0
            total_loss_between_updates = 0
    
    avg_train_loss = total_loss / n_samples
    return avg_train_loss

# %%

criterion = nn.CrossEntropyLoss()
model = RNN(input_size=X_train.shape[-1], hidden_size=8, output_size=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,)
print(model)

# %%

EPOCHS = 200

test_loss, test_acc = test(model, criterion)
print(f"avg test loss = {test_loss:.5f} | accuracy = {test_acc * 100:.2f}%")

logs = []
initial_log = {
    "epoch": 0,
    "test_loss": test_loss,
    "test_acc": test_acc,
    "train_loss": None,
}
logs.append(initial_log)

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
    }
    logs.append(log)

# %%

df_logs = pd.DataFrame(logs).set_index('epoch')
plt.plot(df_logs[['train_loss','test_loss']], label=['train_loss', 'test_loss'])

plt.legend()
# %%

y_train.sum() / len(y_train), y_test.sum() / len(y_test)