import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch import nn

from peds_model.Dataset_Class import AlphaQoIDataset
from peds_model.Downsample_Class import DownsampleAlpha
from peds_model.alpha_layer import AlphaLayer
from peds_model.QoI import QuantityOfInterest
from training_NN import AlphaNN

# Evaluation function for fixed w
def evaluate_fixed_weight_model(w_value, model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for alpha_batch, qoi_batch in test_loader:
            alpha_batch = alpha_batch.float()
            qoi_batch = qoi_batch.float()

            # Get downsampled and NN alpha
            alpha_ds = downsampler(alpha_batch)
            alpha_nn = model(alpha_batch)

            # Fixed interpolation
            alpha_tilde = w_value * alpha_ds + (1 - w_value) * alpha_nn

            # Solve PDE and extract QoI
            u_pred = alpha_layer(alpha_tilde)
            qoi_pred = torch.stack([qoi_extractor(u_i) for u_i in u_pred])

            loss = criterion(qoi_pred, qoi_batch)
            total_loss += loss.item() * len(alpha_batch)
            num_samples += len(alpha_batch)

    return total_loss / num_samples

# Main function
def main():
    # Dataset and loaders
    dataset = AlphaQoIDataset(n_samples=1000, L=1000, n_fibres=10, sigma=0.02, alpha_0=0.0, alpha_1=1.0)
    _, _, test_data = random_split(dataset, [800, 100, 100])
    test_loader = DataLoader(test_data, batch_size=16)

    # Load trained NN model
    model = AlphaNN(input_length=1000, output_length=501)
    model.load_state_dict(torch.load("trained_alphaNN_model.pt"))
    model.eval()

    # Fixed PDE components
    global downsampler, alpha_layer, qoi_extractor
    downsampler = DownsampleAlpha(factor=2)
    alpha_layer = AlphaLayer(f=torch.ones(501, dtype=torch.float32))
    qoi_extractor = QuantityOfInterest([0.25, 0.5, 0.75], x_grid=torch.linspace(0, 1, 501))

    criterion = nn.MSELoss()

    # Sweep weights and collect losses
    weights = torch.linspace(0, 1, steps=11)  # 0.0 to 1.0
    losses = []

    for w in weights:
        loss = evaluate_fixed_weight_model(w.item(), model, test_loader, criterion)
        print(f"w = {w:.2f} → Test Loss = {loss:.6f}")
        losses.append(loss)

    # Plot
    plt.plot(weights.numpy(), losses, marker='o')
    plt.xlabel("Fixed interpolation weight (w)")
    plt.ylabel("Test Loss")
    plt.title("Test Loss vs Fixed Interpolation Weight")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

