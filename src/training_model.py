import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import random_split

import matplotlib.pyplot as plt

# Dataset Class 
from peds_model.Dataset_Class import AlphaQoIDataset 
# Thomas Algorithm 
from peds_model.T_Alg import thomas_algorithm
# Alpha Layer Model 
from peds_model.alpha_layer import AlphaLayer, AlphaFunction
# Quantity of Interst 
from peds_model.QoI import QuantityOfInterest

from measure_error import compute_relative_l2_error





dataset = AlphaQoIDataset(n_samples=1000, L=1000, n_fibres=5, sigma=0.02, alpha_0=0.0, alpha_1=1.0)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])


train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)
test_loader = DataLoader(test_data, batch_size=16)

model = AlphaLayer(f=torch.tensor([1.0], dtype=torch.float64))

# Define CNN 
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaCNN(nn.Module):
    def __init__(self, input_length=1000, output_size=16):  
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(output_size=50)
        self.fc = nn.Linear(8 * 50, output_size)  # maps to 16 QoI outputs

    def forward(self, x):
        x = x.unsqueeze(1)          # [B, 1, L]
        x = F.relu(self.conv1(x))   # [B, 8, L]
        x = self.pool(x)            # [B, 8, 50] (after adaptive pooling)
        x = x.view(x.size(0), -1)   # Flatten to [B, 8*50]
        x = self.fc(x)              # [B, 3] (final QoI prediction)
        return x




model = AlphaCNN(input_length=1000, output_size=3)

criterion = torch.nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

train_losses = []
val_losses = []
rel_errors = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for alpha_batch, qoi_batch in train_loader:
        alpha_batch = alpha_batch.float()
        qoi_batch = qoi_batch.float()

        optimizer.zero_grad()
        qoi_pred = model(alpha_batch)
        loss = criterion(qoi_pred, qoi_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 🔁 Validation after each epoch
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for alpha_batch, qoi_batch in val_loader:
            alpha_batch = alpha_batch.float()
            qoi_batch = qoi_batch.float()
            qoi_pred = model(alpha_batch)
            loss = criterion(qoi_pred, qoi_batch)
            val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

    # 🔁 Compute relative error after each epoch
    rel_error = compute_relative_l2_error(model, test_loader)
    rel_errors.append(rel_error)

    # ✅ Print all metrics
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Relative L2 Error: {rel_error:.6f}")


torch.save(model.state_dict(), "trained_qoi_model.pt")

# Plot training loss and relative L2 error
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 5))

# 📉 Plot Training Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss per Epoch")
plt.grid(True)
plt.legend()

# 📉 Plot Relative L2 Error
plt.subplot(1, 2, 2)
plt.plot(epochs, rel_errors, label="Relative L2 Error", color='orange', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Relative L2 Error")
plt.title("Relative L2 Error per Epoch")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
