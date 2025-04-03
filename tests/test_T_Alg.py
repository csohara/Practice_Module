from peds_model.T_Alg import thomas_algorithm 


import torch
from peds_model.T_Alg import thomas_algorithm

def test_batched_thomas_identity_matrix():
    B = 3  # batch size
    N = 5  # system size

    a = torch.zeros((B, N - 1), dtype=torch.float64)
    b = torch.ones((B, N), dtype=torch.float64)
    c = torch.zeros((B, N - 1), dtype=torch.float64)

    f = torch.tensor([
        [10, 20, 30, 40, 50],
        [1, 2, 3, 4, 5],
        [100, 200, 300, 400, 500],
    ], dtype=torch.float64)

    result = thomas_algorithm(a, b, c, f)

    assert torch.allclose(result, f), f"Expected {f}, got {result}"

