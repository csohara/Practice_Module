
import torch 
DTYPE = torch.float64 

from peds_model.T_Alg import thomas_algorithm

# Matrix Construction 


# Custom Autograd Function 
class AlphaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, metadata, input):
        """
        Forward solve: u = A(alpha)^(-1) f
        change shape to (B,N) for batches 
        """
        f = metadata["f"].to(DTYPE)

        # Make sure f is right shape 
        if f.ndim == 1:
            f = f.unsqueeze(0).expand(input.shape[0], -1)


        a, b, c = AlphaFunction.tridiagonal_matrix(input)
        u = thomas_algorithm(a, b, c, f)
        ctx.save_for_backward(input, a, b, c, u, f)
        return u
    
    @staticmethod
    def tridiagonal_matrix(alpha):
        """
        Constructs the tridiagonal matrix A(alpha) where:
         a_j = alpha_j^2
        b_j = 1 + alpha_j^3
        c_j = alpha_{j+1}^2 + 2*alpha_{j+1}
        """
        alpha = alpha.to(DTYPE)

        # Make sure alpha is right shape
        if alpha.ndim == 1:
            alpha = alpha.unsqueeze(0)


        #N = alpha.shape[0]
        a = alpha[:, :-1] ** 2
        b = 5 + alpha ** 3
        b = torch.clamp(b, min=1e-2)
        c = alpha[:, 1:] ** 2 + 2 * alpha[:, 1:]
        return a, b, c

    @staticmethod
    def backward(ctx, grad_output):
        """
        Batched adjoint gradient:
        grad_input[b, j] = - w_b^T (dA_b/dalpha_bj * u_b)
        where A_b^T w_b = grad_output_b, solved via Thomas algorithm.
        """
        input, a, b, c, u, f = ctx.saved_tensors
        B, N = input.shape

        # Solve Aᵀ w = grad_output for each batch
        w = thomas_algorithm(c, b, a, grad_output.to(DTYPE))  # (B, N)

        grad_input = torch.zeros_like(input, dtype=DTYPE)  # (B, N)

        for batch in range(B):
            alpha_b = input[batch]
            u_b = u[batch]
            w_b = w[batch]

            for j in range(N):
                dA_dalpha_u_j = torch.zeros(N, dtype=DTYPE)

                if j < N - 1:
                    dA_dalpha_u_j[j + 1] += 2 * alpha_b[j] * u_b[j]
                dA_dalpha_u_j[j] += 3 * alpha_b[j]**2 * u_b[j]
                if j > 0:
                    dA_dalpha_u_j[j - 1] += (2 * alpha_b[j] + 2) * u_b[j]

                grad_input[batch, j] = - torch.dot(w_b, dA_dalpha_u_j)

        return None, grad_input
    
class AlphaLayer(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.metadata = dict(f=f.to(DTYPE))
    def forward(self, input):
        return AlphaFunction.apply(self.metadata, input)