import torch as t

def get_one_fv(dim: int, S: list[float]) -> t.Tensor: 
    """
    Returns one feature vector of length dim where each dimension i has 
    associated sparsity S[i] and importance I[i]
    """
    fv = t.rand(dim)
    mask = t.rand(dim) < t.tensor(S)
    fv[mask] = 0
    return fv

def get_n_fvs(n: int, dim: int, S: int) -> t.Tensor: 
    """
    Returns n feature vectors of length dim where each dimension i has
    associated sparsity S
    """
    return t.stack([get_one_fv(dim, [S for _ in range(dim)]) for _ in range(n)])