from peds_model.T_Alg import thomas_algorithm 


import torch

def test_thomas_algorithm_identity_matrix():
    N = 5
    a = torch.zeros(N, dtype=torch.float64)  
    b = torch.ones(N, dtype=torch.float64)   
    c = torch.zeros(N, dtype=torch.float64)  
    f = torch.tensor([10, 20, 30, 40, 50], dtype=torch.float64)

    result = thomas_algorithm(a, b, c, f)

    # Since A is identity, result should equal f
    assert torch.equal(result, f), f"Expected {f}, got {result}"

