# import os
# import numpy as np
# import pandas as pd
# import torch
# from sklearn.preprocessing import StandardScaler

# class SlidingWindowDataset(torch.utils.data.Dataset):
#     """
#     Sliding window dataset with day-of-week feature.
#     Each node has 2 features: [Average Speed, Day-of-week]
#     """
#     def __init__(self, csv_path, seq_len=12, pred_len=1):
#         if not os.path.exists(csv_path):
#             raise FileNotFoundError(f"CSV not found: {csv_path}")

#         df = pd.read_csv(csv_path)

#         # --- Parse date and add day-of-week ---
#         if 'Date' in df.columns:
#             df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#             df['day_of_week'] = df['Date'].dt.dayofweek  # 0=Mon, ..., 6=Sun
#             day_onehot = pd.get_dummies(df['day_of_week'], prefix='dow')
#             df = pd.concat([df, day_onehot], axis=1)

#         # --- Build node names ---
#         if 'Area Name' in df.columns and 'Road/Intersection Name' in df.columns:
#             df['node'] = df['Area Name'].astype(str) + " || " + df['Road/Intersection Name'].astype(str)
#         elif 'Road/Intersection Name' in df.columns:
#             df['node'] = df['Road/Intersection Name'].astype(str)
#         else:
#             raise ValueError("CSV must contain node columns")

#         # --- Pivot tables ---
#         pivot_speed = df.pivot_table(index='Date', columns='node', values=day_onehot.columns.tolist())
#         pivot_speed = pivot_speed.sort_index().ffill().bfill().fillna(0.0)

#         pivot_day = df.pivot_table(index='Date', columns='node', values='day_of_week')
#         pivot_day = pivot_day.sort_index().ffill().bfill().fillna(0.0)

#         # --- Stack features per node ---
#         # values = np.stack([pivot_speed.values, pivot_day.values], axis=-1)  # (T, N, 2)
#         values = np.concatenate([pivot_speed.values[..., np.newaxis], pivot_day.values], axis=-1)  # (T, N, 8)
#         self.in_features = values.shape[-1]
#         self.node_names = list(pivot_speed.columns)
#         self.num_nodes = values.shape[1]

#         # --- Normalize only the Average Speed (first feature) ---
#         scaler = StandardScaler()
#         values[:,:,0] = scaler.fit_transform(values[:,:,0])
#         self.scaler = scaler

#         # --- Sliding windows ---
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         X, Y = [], []
#         max_t = values.shape[0] - seq_len - pred_len + 1
#         for t in range(max_t):
#             X.append(values[t:t+seq_len, :, :])        # (seq_len, N, 2)
#             Y.append(values[t+seq_len:t+seq_len+pred_len, :, 0])  # predict speed only

#         self.X = np.stack(X, axis=0)  # (num_samples, seq_len, N, 2)
#         self.Y = np.stack(Y, axis=0)  # (num_samples, pred_len, N)

#     def __len__(self):
#         return self.X.shape[0]

#     def __getitem__(self, idx):
#         x = torch.from_numpy(self.X[idx]).float()  # (seq_len, N, 2)
#         y = torch.from_numpy(self.Y[idx]).float()  # (pred_len, N)
#         if self.pred_len == 1:
#             y = y.squeeze(0)  # (N,)
#         return x, y

# def build_adjacency(csv_path, threshold=0.25):
#     """Return adjacency matrix (N x N) based on Pearson correlation."""
#     df = pd.read_csv(csv_path)
#     if 'Area Name' in df.columns and 'Road/Intersection Name' in df.columns:
#         df['node'] = df['Area Name'].astype(str) + " || " + df['Road/Intersection Name'].astype(str)
#     elif 'Road/Intersection Name' in df.columns:
#         df['node'] = df['Road/Intersection Name'].astype(str)
#     else:
#         raise ValueError("CSV must contain node columns")

#     if 'Date' in df.columns:
#         df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

#     pivot = df.pivot_table(index='Date', columns='node', values='Average Speed')
#     pivot = pivot.sort_index().ffill().bfill().fillna(0.0)
#     corr = pivot.corr().fillna(0.0)
#     nodes = list(corr.columns)
#     N = len(nodes)

#     adj = torch.zeros((N, N), dtype=torch.float32)
#     for i in range(N):
#         for j in range(N):
#             if i == j:
#                 adj[i, j] = 1.0
#             elif abs(corr.iloc[i, j]) >= threshold:
#                 adj[i, j] = float(abs(corr.iloc[i, j]))
#     if adj.sum() == 0:
#         adj = torch.ones((N, N), dtype=torch.float32)
#         torch.fill_diagonal_(adj, 0.0)

#     return adj, nodes

# def get_dataloaders(csv_path, seq_len=12, pred_len=1, batch_size=8, val_split=0.15):
#     ds = SlidingWindowDataset(csv_path, seq_len=seq_len, pred_len=pred_len)
#     total = len(ds)
#     val_n = int(total * val_split)
#     train_n = total - val_n
#     train_set, val_set = torch.utils.data.random_split(ds, [train_n, val_n])
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
#     val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
#     return train_loader, val_loader, ds.num_nodes, ds.node_names, ds.scaler

import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

class SlidingWindowDataset(torch.utils.data.Dataset):
    """
    Sliding window dataset with day-of-week feature as a single integer per node.
    Each node has 2 features: [Average Speed, Day-of-week]
    """
    def __init__(self, csv_path, seq_len=12, pred_len=1):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # --- Parse date and add day-of-week ---
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['day_of_week'] = df['Date'].dt.dayofweek  # 0=Mon,...6=Sun

        # --- Build node names ---
        if 'Area Name' in df.columns and 'Road/Intersection Name' in df.columns:
            df['node'] = df['Area Name'].astype(str) + " || " + df['Road/Intersection Name'].astype(str)
        elif 'Road/Intersection Name' in df.columns:
            df['node'] = df['Road/Intersection Name'].astype(str)
        else:
            raise ValueError("CSV must contain node columns")

        # --- Pivot tables ---
        pivot_speed = df.pivot_table(index='Date', columns='node', values='Average Speed')
        pivot_speed = pivot_speed.sort_index().ffill().bfill().fillna(0.0)

        pivot_day = df.pivot_table(index='Date', columns='node', values='day_of_week')
        pivot_day = pivot_day.sort_index().ffill().bfill().fillna(0.0)

        # --- Stack features ---
        speed_feat = pivot_speed.values[..., np.newaxis]  # (T, N, 1)
        day_feat   = pivot_day.values[..., np.newaxis]    # (T, N, 1)
        values = np.concatenate([speed_feat, day_feat], axis=-1)  # (T, N, 2)

        self.in_features = values.shape[-1]
        self.node_names = list(pivot_speed.columns)
        self.num_nodes = values.shape[1]

        # --- Normalize only Average Speed ---
        scaler = StandardScaler()
        values[:,:,0] = scaler.fit_transform(values[:,:,0])
        self.scaler = scaler

        # --- Sliding windows ---
        self.seq_len = seq_len
        self.pred_len = pred_len
        X, Y = [], []
        max_t = values.shape[0] - seq_len - pred_len + 1
        for t in range(max_t):
            X.append(values[t:t+seq_len, :, :])                   # (seq_len, N, 2)
            Y.append(values[t+seq_len:t+seq_len+pred_len, :, 0])  # predict speed only

        self.X = np.stack(X, axis=0)  # (num_samples, seq_len, N, 2)
        self.Y = np.stack(Y, axis=0)  # (num_samples, pred_len, N)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()  # (seq_len, N, 2)
        y = torch.from_numpy(self.Y[idx]).float()  # (pred_len, N)
        if self.pred_len == 1:
            y = y.squeeze(0)  # (N,)
        return x, y

def build_adjacency(csv_path, threshold=0.25):
    """Return adjacency matrix (N x N) based on Pearson correlation."""
    df = pd.read_csv(csv_path)
    if 'Area Name' in df.columns and 'Road/Intersection Name' in df.columns:
        df['node'] = df['Area Name'].astype(str) + " || " + df['Road/Intersection Name'].astype(str)
    elif 'Road/Intersection Name' in df.columns:
        df['node'] = df['Road/Intersection Name'].astype(str)
    else:
        raise ValueError("CSV must contain node columns")

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    pivot = df.pivot_table(index='Date', columns='node', values='Average Speed')
    pivot = pivot.sort_index().ffill().bfill().fillna(0.0)
    corr = pivot.corr().fillna(0.0)
    nodes = list(corr.columns)
    N = len(nodes)

    adj = torch.zeros((N, N), dtype=torch.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                adj[i,j] = 1.0
            elif abs(corr.iloc[i,j]) >= threshold:
                adj[i,j] = float(abs(corr.iloc[i,j]))
    if adj.sum() == 0:
        adj = torch.ones((N,N), dtype=torch.float32)
        torch.fill_diagonal_(adj, 0.0)
    return adj, nodes

def get_dataloaders(csv_path, seq_len=12, pred_len=1, batch_size=8, val_split=0.15):
    ds = SlidingWindowDataset(csv_path, seq_len=seq_len, pred_len=pred_len)
    total = len(ds)
    val_n = int(total * val_split)
    train_n = total - val_n
    train_set, val_set = torch.utils.data.random_split(ds, [train_n, val_n])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, ds.num_nodes, ds.node_names, ds.scaler
