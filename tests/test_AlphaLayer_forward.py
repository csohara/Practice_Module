import torch
import numpy as np
from peds_model.alpha_layer import AlphaLayer, AlphaFunction
from peds_model.T_Alg import thomas_algorithm

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

def test_alpha_layer_forward_pass():
    # Test setup
    n = 5
    rng = np.random.default_rng(0)
    alpha = torch.tensor(rng.normal(size=n), dtype=DTYPE)
    f = torch.tensor(rng.normal(size=n), dtype=DTYPE)

    # Run AlphaLayer
    model = AlphaLayer(f)
    u = model(alpha)

    # Manual solve
    a, b, c = AlphaFunction.tridiagonal_matrix(alpha)
    expected_u = thomas_algorithm(a, b, c, f)

    # Check match
    assert torch.allclose(u, expected_u, atol=1e-8), (
        f"AlphaLayer forward failed.\nExpected: {expected_u}\nGot: {u}"
    )
