import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from peds_model.Dataset_Class import AlphaQoIDataset
from peds_model.alpha_layer import AlphaLayer
from peds_model.QoI import QuantityOfInterest

# Define the NN that outputs alphaNN
class AlphaNN(nn.Module):
    def __init__(self, input_length=1000, output_length=501):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding=2) # learns local patterns
        self.pool = nn.AdaptiveAvgPool1d(output_size=output_length) # matches downsample size 
        self.fc = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1)  # Combines features to output alphaNN

    def forward(self, x):
        x = x.unsqueeze(1)              # [B, 1, L] - add channel dim
        x = F.relu(self.conv1(x))       # [B, 8, L] - learn local features
        x = self.pool(x)                # [B, 8, output_length] - reduce resolution
        x = self.fc(x)                  # [B, 1, output_length] - predict alphaNN
        x = x.squeeze(1)                # [B, output_length] - remove channel dim
        return x

# Load dataset
L = 1000
output_length = 501
# dataset gives alpha, QoI pairs
dataset = AlphaQoIDataset(n_samples=1000, L=L, n_fibres=10, sigma=0.02, alpha_0=0.0, alpha_1=1.0)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)
test_loader = DataLoader(test_data, batch_size=16)

# Define the model and optimizer
model = AlphaNN(input_length=L, output_length=output_length)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


# Define physics-based model (used to convert alphaNN into QoI)
#define low res spatial grid 
x_lowres = torch.linspace(0, 1, output_length, dtype=torch.float32)  # Ensure float32
torch_indices_dtype = torch.float32
qoi_extractor = QuantityOfInterest([0.25, 0.5, 0.75], x_grid=x_lowres)
alpha_layer = AlphaLayer(f=torch.ones(output_length, dtype=torch.float32))

full_model = nn.Sequential(model,alpha_layer,qoi_extractor) # combine NN and solver


# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for alpha, qoi_true in train_loader: # takes high res alpha [B, 1000] and qoi [B, 3]
        alpha = alpha.float()
        qoi_true = qoi_true.float()

        optimizer.zero_grad()

        qoi_pred = full_model(alpha)
        #alpha_nn = model(alpha)  # Predict alphaNN by passing to NN (output is [B, 501])
        #qoi_pred = qoi_extractor(u_pred.float())  # Extract QoI from predicted u
        #u_pred = alpha_layer(alpha_nn.float())  # Use predicted alphaNN to solve for u

        loss = criterion(qoi_pred, qoi_true)  # Supervise indirectly through QoI
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for alpha, qoi_true in val_loader:
            alpha = alpha.float()
            qoi_true = qoi_true.float()

            alpha_nn = model(alpha)
            u_pred = alpha_layer(alpha_nn.float())
            qoi_pred = torch.stack([qoi_extractor(u_i.float()) for u_i in u_pred])

            loss = criterion(qoi_pred, qoi_true)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    #print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "trained_alphaNN_model.pt")# Load model and set to eval mode
model.eval()
alpha, _ = dataset[0]  # Get one high-res alpha from dataset
alpha_nn = model(alpha.unsqueeze(0))  # Add batch dimension

# Convert to NumPy 
alpha_nn_np = alpha_nn.squeeze(0).detach().cpu().numpy()
#print("Predicted alphaNN (shape):", alpha_nn_np.shape)
#print("First 10 values:", alpha_nn_np[:10])

