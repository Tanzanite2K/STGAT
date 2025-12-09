



import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import math
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from data_loader.dataloader import get_dataloaders, build_adjacency
from models.st_gaat import STGAAT

# --- Setup ---
CSV_PATH = "dataset/bangalore_traffic.csv"
SEQ_LEN = 12
PRED_LEN = 1
BATCH_SIZE = 32
EPOCHS = 75
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "graphs"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Data ---
train_loader, val_loader, N, node_names, scaler = get_dataloaders(CSV_PATH, SEQ_LEN, PRED_LEN, BATCH_SIZE)
adj, _ = build_adjacency(CSV_PATH)
adj = adj.to(DEVICE)

# --- Model ---
model = STGAAT(num_nodes=N, in_features=2, gat_out=16, lstm_hidden=64, n_pred=PRED_LEN).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training ---
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(x, adj)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = model(x, adj)
            val_loss += criterion(y_pred, y).item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}] → Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# --- FIGURE 1: Training and Validation Loss ---
plt.figure()
plt.plot(train_losses, label="Train Loss", linewidth=2)
plt.plot(val_losses, label="Validation Loss", linewidth=2)
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, "figure1_train_val_loss.png"), dpi=300)
plt.close()
print("Saved figure1_train_val_loss.png")

# --- Evaluation ---
model.eval()
x, y = next(iter(val_loader))
x, y = x.to(DEVICE), y.to(DEVICE)
with torch.no_grad():
    y_pred = model(x, adj)

y_true = y.cpu().numpy()
y_pred = y_pred.cpu().numpy()
day_of_week = x[:, -1, :, 1].cpu().numpy().astype(int)

# --- FIGURE 2: Predicted vs True (Per Node, Colored by Day) ---
num_cols = 4
num_rows = math.ceil(N / num_cols)
fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows), squeeze=False)

day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
cmap = mcolors.ListedColormap(plt.get_cmap('tab10').colors[:7])
norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, 7.5, 1), ncolors=7)

for i in range(N):
    r, c = divmod(i, num_cols)
    ax = axes[r][c]
    ax.scatter(range(len(y_true[:, i])), y_true[:, i],
               c=day_of_week[:, i], cmap=cmap, norm=norm, marker='o', alpha=0.8, label='Actual')
    ax.scatter(range(len(y_pred[:, i])), y_pred[:, i],
               c=day_of_week[:, i], cmap=cmap, norm=norm, marker='x', alpha=0.8, label='Predicted')
    ax.set_title(node_names[i], fontsize=10)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalized Speed")
    ax.grid(True)

for j in range(N, num_rows*num_cols):
    r, c = divmod(j, num_cols)
    fig.delaxes(axes[r][c])

plt.tight_layout(rect=[0, 0, 0.95, 1])
cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                    cax=cbar_ax, ticks=np.arange(0, 7))
cbar.set_ticklabels(day_labels)
cbar.set_label("Day of Week")
plt.savefig(os.path.join(SAVE_DIR, "figure2_pred_vs_true.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure2_pred_vs_true.png")

# --- FIGURE 3: Prediction Error Distribution ---
errors = (y_pred - y_true).flatten()
plt.figure()
plt.hist(errors, bins=50, color='skyblue', edgecolor='black')
plt.title("Prediction Error Distribution")
plt.xlabel("Error (Pred - True)")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, "figure3_error_distribution.png"), dpi=300)
plt.close()
print("Saved figure3_error_distribution.png")

print(" All Figures Saved Successfully in /graphs Folder!")


# --- METRICS ---
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Flatten all values for metric calculation
true_flat = y_true.flatten()
pred_flat = y_pred.flatten()

# MAE
mae = mean_absolute_error(true_flat, pred_flat)

# RMSE
rmse = np.sqrt(mean_squared_error(true_flat, pred_flat))


print("\nEvaluation Metrics:")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")



