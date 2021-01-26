import torch

class MatrixExponential(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        if len(X.shape) == 2:
            assert X.shape[0] == X.shape[1]
            assert (X == X.t()).all(), "X must be symmetric."
            Lam, Q = torch.symeig(X, eigenvectors=True)
            output = Q @ torch.diag(torch.exp(Lam)) @ Q.T
        elif len(X.shape) == 3:
            assert X.shape[1] == X.shape[2]
            assert (X == X.transpose(1, 2)).all(), "X must be symmetric."
            Lam, Q = torch.symeig(X, eigenvectors=True)
            output = torch.bmm(torch.bmm(Q, torch.diag_embed(torch.exp(Lam))), Q.transpose(1, 2))
        else:
            raise ArgumentError("X must be 2 or 3 dimensional.")
        ctx.save_for_backward(X, Lam, Q)
            
        return output

    @staticmethod
    def backward(ctx, S):
        X, Lam, Q = ctx.saved_tensors
        
        if len(X.shape) == 2:
            W = Q.t() @ S @ Q
            tmp = (torch.exp(Lam[None, :] - Lam[:, None]) - 1) / (Lam[None, :] - Lam[:, None])
            tmp[Lam[None, :] == Lam[:, None]] = 1.
            modifier = torch.exp(Lam[:, None]) * tmp
            grad_X = W * modifier
            grad_X = Q @ grad_X @ Q.t()
        elif len(X.shape) == 3:
            W = torch.bmm(torch.bmm(Q.transpose(1, 2), S), Q)
            tmp = (torch.exp(Lam[:, None, :] - Lam[:, :, None]) - 1) / (Lam[:, None, :] - Lam[:, :, None])
            tmp[Lam[:, None, :] == Lam[:, :, None]] = 1.
            modifier = torch.exp(Lam[:, :, None]) * tmp
            grad_X = W * modifier
            grad_X = torch.bmm(torch.bmm(Q, grad_X), Q.transpose(1, 2))
        return grad_X

expm = MatrixExponential().apply