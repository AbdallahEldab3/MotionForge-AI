import torch
import numpy as np
import torch.nn as nn

# --- Load normalization stats (from your preprocessing) ---
X_mean = np.load(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\X_mean.npy")
X_std = np.load(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\X_std.npy")
Y_mean = np.load(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\Y_mean.npy")
Y_std = np.load(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\Y_std.npy")

# --- Define the model exactly as before ---
class MotionTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim_feedforward)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(dim_feedforward, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.output_proj(x)
        return x

# --- Setup device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dimensions (from your training preprocessing) ---
input_dim = 33*3*2 + 3  # local_motion + local_vel + root_vel
output_dim = 33*3       # local_motion prediction

# --- Load model ---
model = MotionTransformer(input_dim=input_dim, output_dim=output_dim).to(device)
model.load_state_dict(torch.load(r"C:\Users\IBRA\project_maham\models\motion_transformer_00.pth", map_location=device))
model.eval()
print("Model loaded and ready for test inference.")

# --- Load a test sequence (just one window for testing) ---
X_seed = np.load(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\X_norm.npy")  # shape (1, seq_len, input_dim)
X_seed = torch.tensor(X_seed, dtype=torch.float32).to(device)

# --- Single forward pass to see output ---
with torch.no_grad():
    Y_pred = model(X_seed)  # (1, seq_len, output_dim)

# --- Denormalize to original scale ---
Y_pred_np = Y_pred.cpu().numpy() * Y_std + Y_mean

print("Test output shape:", Y_pred_np.shape)
print("First frame sample:", Y_pred_np[0, 0, :10])

np.save(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\Y_pred.npy", Y_pred_np)