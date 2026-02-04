import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Load normalization stats (from your preprocessing) ---
Y_true_path = np.load(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\Y_norm.npy")
Y_pred = np.load(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\Y_pred.npy")
Y_std = np.load(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\Y_std.npy")
Y_mean = np.load(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\Y_mean.npy")

Y_true = Y_true_path * Y_std + Y_mean
#Y_pred = Y_pred_path * Y_std + Y_mean
POSE_CONNECTIONS = [
    (11, 13), (13, 15),        # Left arm
    (12, 14), (14, 16),        # Right arm
    (11, 12),                  # Shoulders
    (11, 23), (12, 24),        # Torso
    (23, 24),                  # Hips
    (23, 25), (25, 27),        # Left leg
    (24, 26), (26, 28),        # Right leg
]

seq_id = 0

Y_true = Y_true[seq_id].reshape(-1, 33, 3)
Y_pred = Y_pred[seq_id].reshape(-1, 33, 3)

root = (Y_true[:,23] + Y_true[:,24]) / 2
Y_true = Y_true - root[:, None, :]
Y_pred = Y_pred - root[:, None, :]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
titles = ["Ground Truth", "Model Prediction"]

for ax, title in zip(axes, titles):
    ax.set_title(title)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

lines_true = [axes[0].plot([], [], 'o-', lw=2)[0] for _ in POSE_CONNECTIONS]
lines_pred = [axes[1].plot([], [], 'o-', lw=2)[0] for _ in POSE_CONNECTIONS]

def update(frame):
    for i, (a, b) in enumerate(POSE_CONNECTIONS):
        # Ground truth
        x = [Y_true[frame, a, 0], Y_true[frame, b, 0]]
        y = [Y_true[frame, a, 1], Y_true[frame, b, 1]]
        lines_true[i].set_data(x, y)

        # Prediction
        x = [Y_pred[frame, a, 0], Y_pred[frame, b, 0]]
        y = [Y_pred[frame, a, 1], Y_pred[frame, b, 1]]
        lines_pred[i].set_data(x, y)

    return lines_true + lines_pred

ani = FuncAnimation(fig, update, frames=Y_true.shape[0], interval=50)
plt.show()