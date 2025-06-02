#import torch
#import torch.nn as nn

#from peds_model.Downsample_Class import DownsampleAlpha
#from training_NN import AlphaNN
#from peds_model.alpha_layer import AlphaLayer
#from peds_model.QoI import QuantityOfInterest

#class HybridAlphaModel(nn.Module):
    #def __init__(self, input_length=1000, downsample_factor=2, qoi_points=[0.25, 0.5, 0.75]):
        #super().__init__()

        # Input
        #self.downsampler = DownsampleAlpha(factor=downsample_factor)
        #self.alpha_nn = AlphaNN(input_length=input_length, output_length=(input_length // downsample_factor + 1))

        # PDE solver
        #self.alpha_layer = AlphaLayer(f=torch.ones(input_length // downsample_factor + 1, dtype=torch.float32))

        # Quantity of Interest extractor
        #x_grid_downsampled = torch.linspace(0, 1, input_length // downsample_factor + 1)
        #self.qoi_extractor = QuantityOfInterest(qoi_points, x_grid=x_grid_downsampled)

    #def forward(self, alpha_highres):
        #if alpha_highres.ndim == 1:
            #alpha_highres = alpha_highres.unsqueeze(0)  # Ensure batch dim

        # Compute downsampled and NN-based alpha **independently**
        #alpha_ds = self.downsampler(alpha_highres)       # [B, L_ds]
        #alpha_nn = self.alpha_nn(alpha_highres)          # [B, L_ds]

        # Combine to get corrected alpha (alpha tilde)
        #alpha_tilde = 0.5 * (alpha_ds + alpha_nn)        # [B, L_ds]

        # Solve PDE and extract QoI
        #u_pred = self.alpha_layer(alpha_tilde)           # [B, L_ds]
        #qoi = torch.stack([self.qoi_extractor(u_i) for u_i in u_pred])  # [B, num_qoi]

        #return qoi
import torch
import torch.nn as nn

from peds_model.Downsample_Class import DownsampleAlpha
from training_NN import AlphaNN
from peds_model.alpha_layer import AlphaLayer
from peds_model.QoI import QuantityOfInterest

class HybridAlphaModel(nn.Module):
    def __init__(self, input_length=1000, downsample_factor=2, qoi_points=[0.25, 0.5, 0.75]):
        super().__init__()

        # Input
        self.downsampler = DownsampleAlpha(factor=downsample_factor)
        self.alpha_nn = AlphaNN(input_length=input_length, output_length=(input_length // downsample_factor + 1))

        # Learnable weighting parameter
        self.raw_w = nn.Parameter(torch.tensor(0.5))

        # PDE solver
        self.alpha_layer = AlphaLayer(f=torch.ones(input_length // downsample_factor + 1, dtype=torch.float32))

        # Quantity of Interest extractor
        x_grid_downsampled = torch.linspace(0, 1, input_length // downsample_factor + 1)
        self.qoi_extractor = QuantityOfInterest(qoi_points, x_grid=x_grid_downsampled)

    def forward(self, alpha_highres):
        if alpha_highres.ndim == 1: # ensure batch size correct 
            alpha_highres = alpha_highres.unsqueeze(0)  

        # Compute downsampled and NN-based alpha 
        alpha_ds = self.downsampler(alpha_highres)       # [B, ds]
        alpha_nn = self.alpha_nn(alpha_highres)          # [B, ds]

        # (0, 1) using sigmoid
        w = torch.sigmoid(self.raw_w)
        #print(f"Current learned weight (w): {w.item():.4f}")  
        # Combine to get corrected alpha (alpha tilde)
        alpha_tilde = w * alpha_ds + (1 - w) * alpha_nn  # [B, ds]

        # Solve PDE and extract QoI
        u_pred = self.alpha_layer(alpha_tilde)           # [B, ds]
        qoi = torch.stack([self.qoi_extractor(u_i) for u_i in u_pred])  # [B, num_qoi]

        return qoi

    def get_weight(self):
        return torch.sigmoid(self.raw_w).item()