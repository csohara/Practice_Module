import torch
import numpy as np
import matplotlib.pyplot as plt
    
import torch
from torch.utils.data import Dataset
import numpy as np

def alpha_step_function(x, regions, alpha_0, alpha_1):
    # Example implementation; replace with yours
    alpha = np.full_like(x, alpha_0)
    for (start, end) in regions:
        alpha[(x >= start) & (x <= end)] = alpha_1
    return alpha

class AlphaFieldDataset(Dataset):
    def __init__(self, n_samples=100, L=1000, n_fibres=5, sigma=0.02, alpha_0=0.0, alpha_1=1.0):
        """
        Dataset class to generate multiple high-resolution alpha fields.
        """
        self.n_samples = n_samples
        self.L = L
        self.n_fibres = n_fibres
        self.sigma = sigma
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1

        self.x = np.linspace(0, 1, L)  # Spatial grid
        self.samples = [self._generate_sample() for _ in range(n_samples)]

    def _generate_sample(self):
        """
        Generate a single high-res alpha field with non-overlapping fibres.
        """
        while True:
            centres = np.sort(np.random.uniform(0, 1, self.n_fibres))
            if centres[0] < self.sigma or centres[-1] > 1 - self.sigma:
                continue
            if np.all(np.diff(centres) >= 2 * self.sigma):
                break
        fibre_regions = [(c - self.sigma, c + self.sigma) for c in centres]
        alpha_np = alpha_step_function(self.x, fibre_regions, self.alpha_0, self.alpha_1)
        alpha_tensor = torch.tensor(alpha_np, dtype=torch.float64)
        return alpha_tensor, fibre_regions

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return self.n_samples