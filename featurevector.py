import torch

def get_one_fv(dim: int, S: list[float]) -> torch.Tensor: 
    """
    Returns one feature vector of length dim where each dimension i has 
    associated sparsity S[i] and importance I[i]
    """
    fv = torch.rand(dim)
    mask = torch.rand(dim) < torch.tensor(S)
    fv[mask] = 0
    return fv

def get_n_fvs(n: int, dim: int, S: int) -> torch.Tensor: 
    """
    Returns n feature vectors of length dim where each dimension i has
    associated sparsity S
    """
    return torch.stack([get_one_fv(dim, [S for _ in range(dim)]) for _ in range(n)])