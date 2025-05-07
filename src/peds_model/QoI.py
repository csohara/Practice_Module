import torch

class QuantityOfInterest:
    def __init__(self, sample_points, x_grid):
        self.sample_points = sample_points
        self.x_grid = torch.tensor(x_grid, dtype=torch.float64)
        self.indices = self._compute_indices()

    def _compute_indices(self):
        indices = []
        for xi in self.sample_points:
            idx = (torch.abs(self.x_grid - xi)).argmin().item()
            indices.append(idx)
        return indices

    def __call__(self, u):
        return u[self.indices]
