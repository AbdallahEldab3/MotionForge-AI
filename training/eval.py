import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------------
# Load normalization stats
# ---------------------------
X_mean = np.load(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\X_mean.npy")
X_std  = np.load(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\X_std.npy")
Y_mean = np.load(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\Y_mean.npy")
Y_std  = np.load(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\Y_std.npy")

# ---------------------------
# Load new video keypoints
# ---------------------------
X_new = np.load(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\X_norm.npy")
Y_true_norm = np.load(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\Y_norm.npy") # optional

# Normalize input
X_new_norm = (X_new - X_mean) / X_std
X_new_tensor = torch.tensor(X_new_norm, dtype=torch.float32)

# ---------------------------
# Define model class
# ---------------------------
class FullMotionTransformer(nn.Module):
    def __init__(self, input_dim, output_dim_local, output_dim_root, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim_feedforward)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # TWO heads
        self.local_head = nn.Linear(dim_feedforward, output_dim_local)
        self.root_head = nn.Linear(dim_feedforward, output_dim_root)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        local_out = self.local_head(x)
        root_out = self.root_head(x)
        return local_out, root_out
# ---------------------------
# Load trained model
# ---------------------------
input_dim = X_new.shape[-1]
output_dim_local = 198        # local motion
output_dim_root = 3            # root motion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FullMotionTransformer(input_dim, output_dim_local, output_dim_root).to(device)
model.load_state_dict(torch.load(r"C:\Users\IBRA\project_maham\models\full_motion_transformer.pth", map_location=device))
model.eval()
print("Model loaded and ready for inference.")

# ---------------------------
# Run inference
# ---------------------------
with torch.no_grad():
    Y_local_pred, Y_root_pred = model(X_new_tensor.to(device))
    Y_local_pred = Y_local_pred.cpu().numpy()
    Y_root_pred = Y_root_pred.cpu().numpy()


# Denormalize predictions
Y_pred_np = np.copy(Y_local_pred)
Y_pred_np[..., :3] += Y_root_pred[:, :, np.newaxis]  # add root to hips

Y_pred_np = Y_pred_np.reshape(-1, 33, 3)

# Ground truth
Y_true = Y_true_norm.reshape(-1, 33, 3)
# ---------------------------
# Visualization
# ---------------------------
POSE_CONNECTIONS = [
    (11, 13), (13, 15),        # Left arm
    (12, 14), (14, 16),        # Right arm
    (11, 12),                  # Shoulders
    (11, 23), (12, 24),        # Torso
    (23, 24),                  # Hips
    (23, 25), (25, 27),        # Left leg
    (24, 26), (26, 28),        # Right leg
]

seq_len = Y_pred_np.shape[0]
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
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
        x = [Y_pred_np[frame, a, 0], Y_pred_np[frame, b, 0]]
        y = [Y_pred_np[frame, a, 1], Y_pred_np[frame, b, 1]]
        lines_pred[i].set_data(x, y)

    return lines_true + lines_pred

ani = FuncAnimation(fig, update, frames=seq_len, interval=50)
plt.show()
