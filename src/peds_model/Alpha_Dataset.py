import torch
import numpy as np
import matplotlib.pyplot as plt
def alpha_step_function(x, fibre_regions, alpha_0=0.0, alpha_1=1.0):
    """
    Create a step-function-like alpha field with fibre regions set to alpha_1,
    and the rest of the domain set to alpha_0.

    Args:
        x (np.ndarray): 1D spatial grid (e.g., linspace from 0 to 1).
        fibre_regions (list of tuples): Each tuple is (start, end) of a fibre.
        alpha_0 (float): Background value.
        alpha_1 (float): Fibre region value.

    Returns:
        np.ndarray: Alpha field of same shape as x.
    """
    alpha = np.full_like(x, alpha_0)
    for (start, end) in fibre_regions:
        alpha[(x >= start) & (x <= end)] = alpha_1
    return alpha

class AlphaFieldDataset:
    def __init__(self, n_samples=100, L=1000, n_fibres=5, sigma=0.02, alpha_0=0.0, alpha_1=1.0):
        """
        Dataset class to generate multiple high-resolution alpha fields.

        Args:
            n_samples (int): Number of alpha fields to generate.
            L (int): Number of spatial points (resolution).
            n_fibres (int): Number of fibre regions per sample.
            sigma (float): Half-width of each fibre.
            alpha_0 (float): Background alpha value.
            alpha_1 (float): Fibre alpha value.
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

        Returns:
            alpha (torch.Tensor): shape [L]
            fibre_regions (list of (start, end))
        """
        # Randomly pick fibre centres with non-overlapping constraint
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
        """
        Get the idx-th sample.

        Returns:
            (alpha: torch.Tensor, fibre_regions: list of tuples)
        """
        return self.samples[idx]

    def __len__(self):
        return self.n_samples
    
    