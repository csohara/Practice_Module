import torch
from torch.utils.data import Dataset
from peds_model.Alpha_Dataset import AlphaFieldDataset
from peds_model.alpha_layer import AlphaLayer
from peds_model.QoI import QuantityOfInterest

class AlphaQoIDataset(Dataset):
    def __init__(self, n_samples=100, L=1000, qoi_points=None, **alpha_kwargs):
        """
        Dataset returning (alpha, QoI(u)) pairs.

        Args:
            n_samples (int): Number of samples.
            L (int): Spatial resolution.
            qoi_points (list): Sample points for QoI (e.g. [0.25, 0.5, 0.75])
            alpha_kwargs: Passed to AlphaFieldDataset (e.g., n_fibres, sigma)
        """
        self.alpha_dataset = AlphaFieldDataset(n_samples=n_samples, L=L, **alpha_kwargs)
        self.x = self.alpha_dataset.x
        self.qoi = QuantityOfInterest(qoi_points or [0.25, 0.5, 0.75], x_grid=self.x)
        self.model = AlphaLayer(f=torch.ones(L, dtype=torch.float64))

    def __len__(self):
        return len(self.alpha_dataset)

    def __getitem__(self, idx):
        alpha, _ = self.alpha_dataset[idx]  # [L]
        alpha_batched = alpha.unsqueeze(0)  # [1, L]
        u = self.model(alpha_batched).squeeze(0)  # [L]
        qoi_u = self.qoi(u)  # [m]
        return alpha, qoi_u


# Create the dataset
dataset = AlphaQoIDataset(n_samples=5, n_fibres=3, sigma=0.02, alpha_1=1.0)

# Collect and print a few (alpha, QoI) pairs
for i in range(len(dataset)):
    alpha, qoi = dataset[i]
    print(f"\nSample {i}")
    print("  alpha (first 5 values):", alpha[:5].numpy())
    print("  QoI:", qoi.numpy())

