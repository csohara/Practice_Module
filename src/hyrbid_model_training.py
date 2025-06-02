import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from peds_model.Dataset_Class import AlphaQoIDataset
from main_hybrid import HybridAlphaModel  

# Hyperparameters
L = 1000
output_length = L // 2 + 1
batch_size = 16
num_epochs = 1
learning_rate = 0.001

# Load dataset and split
full_dataset = AlphaQoIDataset(n_samples=1000, L=L, n_fibres=10, sigma=0.02, alpha_0=0.0, alpha_1=1.0)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

# Instantiate model
model = HybridAlphaModel(input_length=L, downsample_factor=2, qoi_points=[0.25, 0.5, 0.75])

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for alpha_batch, qoi_batch in train_loader:
        alpha_batch = alpha_batch.float()
        qoi_batch = qoi_batch.float()

        optimizer.zero_grad()
        qoi_pred = model(alpha_batch)
        loss = criterion(qoi_pred, qoi_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * alpha_batch.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for alpha_batch, qoi_batch in val_loader:
            alpha_batch = alpha_batch.float()
            qoi_batch = qoi_batch.float()
            qoi_pred = model(alpha_batch)
            loss = criterion(qoi_pred, qoi_batch)
            val_loss += loss.item() * alpha_batch.size(0)

    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

final_w = model.get_weight()
print(f"\nFinal learned weight (w): {final_w:.4f}")
# Save model 
torch.save(model.state_dict(), "trained_hybrid_model.pt")