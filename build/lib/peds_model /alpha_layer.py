
import torch 
DTYPE = torch.float64 

class AlphaLayer(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.metadata = dict(f=f.to(DTYPE))
    def forward(self, input):
        return AlphaFunction.apply(self.metadata, input)