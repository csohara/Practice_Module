from peds_model.alpha_layer import AlphaLayer
from peds_model.Alpha_Dataset import AlphaFieldDataset
import torch

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

# Step 1: Create the dataset of alpha fields
dataset = AlphaFieldDataset(n_samples=3, L=1000)

# Step 2: Extract alpha fields and form batch
alpha_batch = torch.stack([alpha for alpha, _ in dataset])  # [3, 1000]
alpha_batch.requires_grad_()

# Step 3: Define fixed right-hand-side f
f = torch.ones(1000, dtype=DTYPE)

# Step 4: Solve for true u using AlphaLayer
model = AlphaLayer(f)
u_batch = model(alpha_batch)  # [3, 1000]

# Optional: detach and convert to NumPy for inspection or saving
alpha_np = alpha_batch.detach().numpy()
u_np = u_batch.detach().numpy()

# Display result
for i in range(len(dataset)):
    print(f"\nSample {i}:")
    print(f"  alpha: {alpha_np[i, :5]} ...")
    print(f"  u     : {u_np[i, :5]} ...")
    print(f"  alpha max: {alpha_np[i].max()}")  # should be 1.0 if fibres exist

