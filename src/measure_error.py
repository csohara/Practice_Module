# measure_error.py

import torch

def compute_relative_l2_error(model, data_loader, device="cpu"):
    """
    Computes the relative L2 error:
        ||y_pred - y_true||_2 / ||y_true||_2
    over all batches in the given data_loader.
    """
    model.eval()
    numerator = 0.0
    denominator = 0.0

    with torch.no_grad():
        for alpha, qoi_true in data_loader:
            alpha = alpha.to(device).float()
            qoi_true = qoi_true.to(device).float()
            qoi_pred = model(alpha)
            numerator += torch.norm(qoi_pred - qoi_true, p=2).pow(2).item()
            denominator += torch.norm(qoi_true, p=2).pow(2).item()

    return (numerator / denominator) ** 0.5
