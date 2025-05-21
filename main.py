# %%

import pandas as pd
import matplotlib.pyplot as plt
import os

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
df = df[['open','high','low','close']]

# %%

df_o_transformed = df[['open']].div(df['open'].shift(), axis=0) - 1
df_hlc_transformed = df[['high','low','close']].div(df['open'], axis=0) - 1
df_X = pd.merge(df_o_transformed, df_hlc_transformed, left_index=True, right_index=True).dropna()
df_y = ((df[['open']].shift(-1) - df[['open']]).dropna() >= 0).astype(int).rename(columns={'open':'label'})
df_Xy = pd.merge(df_X, df_y, left_index=True, right_index=True)

df_Xy