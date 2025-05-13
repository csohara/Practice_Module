import torch
from torch.utils.data import DataLoader
from peds_model.Dataset_Class import AlphaQoIDataset
from training_model import AlphaCNN  

# Create the dataset and DataLoader
dataset = AlphaQoIDataset(n_samples=1000, L=1000, n_fibres=5, sigma=0.02, alpha_0=0.0, alpha_1=1.0)
_, _, test_data = torch.utils.data.random_split(dataset, [800, 100, 100])  # 80/10/10 split
test_loader = DataLoader(test_data, batch_size=16)

# Recreate the model and load weights
model = AlphaCNN(input_length=1000, output_size=3)
state_dict = torch.load("trained_alpha_model.pt", weights_only=True)
model.load_state_dict(state_dict)

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
