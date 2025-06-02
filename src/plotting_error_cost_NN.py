import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from peds_model.Dataset_Class import AlphaQoIDataset
from measure_error import compute_relative_l2_error

# Define CNN
class AlphaCNN(nn.Module):
    def __init__(self, input_length=1000, output_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=5, padding=2)
        reduced_size = max(10, L // 20)  #
        self.pool = nn.AdaptiveAvgPool1d(output_size=reduced_size)
        self.fc = nn.Linear(8 * reduced_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, L]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# List of discretisations to try
L_values = [100, 150, 250, 400, 600, 800, 1000]
num_epochs = 10

runtime_list = []
error_list = []

for L in L_values:
    print(f"\n--- Running for L = {L} ---")
    
    # Dataset
    dataset = AlphaQoIDataset(n_samples=1000, L=L, n_fibres=5, sigma=0.02, alpha_0=0.0, alpha_1=1.0)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    test_loader = DataLoader(test_data, batch_size=16)

    # Model
    model = AlphaCNN(input_length=L, output_size=3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train for fixed number of epochs
    model.train()
    for epoch in range(num_epochs):
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
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # Evaluate runtime and error
    model.eval()
    start_time = time.perf_counter()
    rel_error = compute_relative_l2_error(model, test_loader)
    end_time = time.perf_counter()
    runtime = end_time - start_time

    runtime_list.append(runtime)
    error_list.append(rel_error)
    print(f"Relative L2 Error: {rel_error:.6f}, Runtime: {runtime:.4f} seconds")

# Plot Error vs Runtime
plt.figure(figsize=(6, 5))
plt.plot(runtime_list, error_list, 'o-', label="Discretisation Sweep")
for i, L in enumerate(L_values):
    plt.text(runtime_list[i], error_list[i], f"L={L}", fontsize=8, ha='right')

plt.xlabel("Runtime (s)")
plt.ylabel("Relative L2 Error")
plt.title("Error vs Runtime across Discretisations")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
