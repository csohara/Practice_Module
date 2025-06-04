import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsampleAlpha(nn.Module):
    def __init__(self, factor=2, mode="replicate"): # replicate is padding mode (repeats edge value)
        super().__init__()
        self.factor = factor
        self.mode = mode

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Downsample a 1D alpha field using average pooling with padding.
        Input shape: [B, L] or [L]
        Output shape: [B, L//factor + 1] or [L//factor + 1]
        """
        was_1d = alpha.ndim == 1 # checks if input is 1D
        if was_1d:
            alpha = alpha.unsqueeze(0)  # add batch dim if 1D

        pad = self.factor // 2
        alpha = alpha.unsqueeze(1)  # shape [B, 1, L]
        alpha_padded = F.pad(alpha, pad=(pad, pad), mode=self.mode)
        alpha_down = F.avg_pool1d(alpha_padded, kernel_size=self.factor, stride=self.factor)
        alpha_down = alpha_down.squeeze(1)  # shape [B, L_down] removes channel dimension

        if was_1d:
            alpha_down = alpha_down.squeeze(0)  # back to [L_down]

        return alpha_down

downsampler = DownsampleAlpha(factor=2)
from peds_model.Alpha_Dataset import AlphaFieldDataset

dataset = AlphaFieldDataset(n_samples=1, L=1000, n_fibres=3, sigma=0.015, alpha_1=1.0)

alpha, _ = dataset[0]
alpha_down = downsampler(alpha)  # shape [501]

#print("Original alpha shape:", alpha.shape)
#print("Downsampled alpha shape:", alpha_down.shape)
#print("First 10 downsampled values:", alpha_down[:10].tolist())