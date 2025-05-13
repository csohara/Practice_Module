import torch

class QuantityOfInterest(torch.nn.Module):
    def __init__(self, sample_points, x_grid):

        super().__init__()
        self.sample_points = sample_points # where we want to sample 
        self.x_grid = torch.tensor(x_grid, dtype=torch.float64) # converts spatial grid into pytorch tensor 
        self.indices = self._compute_indices()

    def _compute_indices(self):
        indices = []
        for xi in self.sample_points:
            idx = (torch.abs(self.x_grid - xi)).argmin().item() # finds index of smallest distane (grid point closest)
            indices.append(idx)
        return indices

    def forward(self, u):
        return u[self.indices]