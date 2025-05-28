from peds_model.Alpha_Dataset import AlphaFieldDataset, alpha_step_function
from peds_model.T_Alg import thomas_algorithm
import torch
import matplotlib.pyplot as plt
from peds_model.QoI import QuantityOfInterest 
from peds_model.alpha_layer import AlphaLayer

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)



from peds_model.QoI import QuantityOfInterest

# Generate 3 samples of input, each 1000 points.
dataset = AlphaFieldDataset(n_samples=3, L=1000, n_fibres=3, sigma=0.02, alpha_1=1.0)
x = dataset.x

# Stack into batched tensor [3, 1000] so we can solve simultaneosly 
alpha = torch.stack([alpha for alpha, _ in dataset])  # shape [3, 1000]
f = torch.ones(1000, dtype=DTYPE)  # constant RHS
model = AlphaLayer(f)

# Forward pass using AlphaLayer
u = model(alpha)  # shape [3, 1000]

# Apply Quantity of Interest
qoi = QuantityOfInterest(sample_points=[0.25, 0.5, 0.75], x_grid=x)
for i in range(alpha.shape[0]):
    qoi_i = qoi(u[i])
    print(f"\nSample {i}")
    print("QoI:", qoi_i.numpy())


plt.plot(x, alpha[0].numpy(), label="alpha")
plt.plot(x, u[0].detach().numpy(), label="u")
plt.legend()
plt.show()




