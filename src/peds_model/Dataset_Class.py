import torch
from torch.utils.data import Dataset
from peds_model.Alpha_Dataset import AlphaFieldDataset #generate alpha fields
from peds_model.alpha_layer import AlphaLayer # solver 
from peds_model.QoI import QuantityOfInterest

class AlphaQoIDataset(Dataset):
    def __init__(self, n_samples=100, L=1000, qoi_points=None, **alpha_kwargs):
        """
        Dataset returning (alpha, QoI(u)) pairs.
        """
        self.alpha_dataset = AlphaFieldDataset(n_samples=n_samples, L=L, **alpha_kwargs) # creates alpha dataset
        self.x = self.alpha_dataset.x # gets spatial grid used to generate alpha 
        self.qoi = QuantityOfInterest(qoi_points or [0.25, 0.5, 0.75], x_grid=self.x)
        self.model = AlphaLayer(f=torch.ones(L, dtype=torch.float64))

    def __len__(self): # gives number of samples 
        return len(self.alpha_dataset)

    
    def __getitem__(self, idx):
        alpha, _ = self.alpha_dataset[idx]
        alpha_batched = alpha.unsqueeze(0)
        u = self.model(alpha_batched).squeeze(0)
        qoi_u = self.qoi(u)

        alpha_tensor = alpha.to(torch.float32)
        qoi_tensor = torch.as_tensor(qoi_u, dtype=torch.float32)  # <- this should be a flat 1D tensor
       

        return alpha_tensor, qoi_tensor




# Create the dataset
dataset = AlphaQoIDataset(n_samples=5, n_fibres=3, sigma=0.02, alpha_1=1.0)

# Collect and print a few (alpha, QoI) pairs
for i in range(len(dataset)):
    alpha, qoi = dataset[i]
    print(f"\nSample {i}")
    print("  alpha (first 5 values):", alpha[:5].numpy())
    print("  QoI:", qoi.numpy())

