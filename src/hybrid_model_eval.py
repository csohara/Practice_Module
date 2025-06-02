import torch
from torch.utils.data import DataLoader, random_split
from peds_model.Dataset_Class import AlphaQoIDataset
from main_hybrid import HybridAlphaModel

# Create dataset and split
dataset = AlphaQoIDataset(n_samples=1000, L=1000, n_fibres=5, sigma=0.02, alpha_0=0.0, alpha_1=1.0)
_, _, test_data = random_split(dataset, [800, 100, 100])
test_loader = DataLoader(test_data, batch_size=16)

# Instantiate model
model = HybridAlphaModel(input_length=1000, downsample_factor=2, qoi_points=[0.25, 0.5, 0.75])


model.eval()

# Define loss
criterion = torch.nn.MSELoss()

# Evaluate on test set
test_loss = 0.0
num_samples = 0
with torch.no_grad():
    for alpha_batch, qoi_batch in test_loader:
        alpha_batch = alpha_batch.float()
        qoi_batch = qoi_batch.float()
        
        predictions = model(alpha_batch)  
        loss = criterion(predictions, qoi_batch)

        test_loss += loss.item() * len(alpha_batch)
        num_samples += len(alpha_batch)

avg_test_loss = test_loss / num_samples
print(f"Test Loss: {avg_test_loss:.4f}")
