import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

X_norm = np.load(r"C:\Users\IBRA\project_maham\data\inference\X_norm.npy")
Y_norm = np.load(r"C:\Users\IBRA\project_maham\data\inference\X_norm.npy")

Y_local = Y_norm[:, :, :-3]
Y_root = Y_norm[:, :, -3:]


class FullMotionDataset(Dataset):
    def __init__(self, X, Y_local, Y_root):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y_local = torch.tensor(Y_local, dtype=torch.float32)
        self.Y_root = torch.tensor(Y_root, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y_local[idx], self.Y_root[idx]

dataset = FullMotionDataset(X_norm, Y_local, Y_root)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    
class FullMotionTransformer(nn.Module):
    def __init__(self, input_dim, local_out_dim, root_out_dim, nhead = 4, num_layers = 2, dim_feedforward = 256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim_feedforward)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model= dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first= True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)
        
        self.local_head = nn.Linear(dim_feedforward, local_out_dim)
        self.root_head = nn.Linear(dim_feedforward, root_out_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.local_head(x), self.root_head(x)
    

input_dim = X_norm.shape[-1]
local_out_dim = Y_local.shape[-1]
root_out_dim = Y_root.shape[-1]

model = FullMotionTransformer(input_dim, local_out_dim, root_out_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


epochs = 200
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_batch, y_local_batch, y_root_batch in dataloader:
        x_batch = x_batch.to(device)
        y_local_batch = y_local_batch.to(device)
        y_root_batch = y_root_batch.to(device)

        optimizer.zero_grad()
        pred_local, pred_root = model(x_batch)

        loss = criterion(pred_local, y_local_batch) + criterion(pred_root, y_root_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(dataset):.6f}")

torch.save(model.state_dict(), r"C:\Users\IBRA\project_maham\models\full_motion_transformer.pth")
print("Training complete and saved.")