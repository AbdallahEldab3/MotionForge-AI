import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

X = np.load(r"C:\Users\IBRA\project_maham\data\inference\X_norm.npy")
Y = np.load(r"C:\Users\IBRA\project_maham\data\inference\Y_norm.npy")
Y_std= np.load(r"C:\Users\IBRA\project_maham\data\inference\Y_std.npy")
Y_mean = np.load(r"C:\Users\IBRA\project_maham\data\inference\Y_mean.npy")

def motion_loss(Y_pred, Y_true, lambda_vel=0.5, lambda_acc=0.1):
    pos_loss = torch.mean((Y_pred - Y_true) ** 2)

    vel_pred = Y_pred[:, 1:] - Y_pred[:, :-1]
    vel_true = Y_true[:, 1:] - Y_true[:, :-1]
    vel_loss = torch.mean((vel_pred - vel_true) ** 2)

    acc_pred = vel_pred[:, 1:] - vel_pred[:, :-1]
    acc_true = vel_true[:, 1:] - vel_true[:, :-1]
    acc_loss = torch.mean((acc_pred - acc_true) ** 2)

    return pos_loss + lambda_vel * vel_loss + lambda_acc * acc_loss



class MotionDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
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
 
 

input_dim = X.shape[-1]
output_dim = Y.shape[-1]
dataset = MotionDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = MotionTransformer(input_dim=input_dim, output_dim=output_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 370
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_batch)

        loss = motion_loss(y_pred, y_batch, lambda_vel=0.5, lambda_acc=0.1)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)

    avg_loss = total_loss / len(dataset)
    print(f"epoch [{epoch+1}/{epochs}] loss: {avg_loss:.6f}")


torch.save(model.state_dict(), r"C:\Users\IBRA\project_maham\models\motion_transformer_00.pth")
print("done...")
