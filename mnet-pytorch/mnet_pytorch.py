from einops import rearrange
import torch

def expand_and_shrink(I, E, F, S, m):
    b, n, d = I.shape
    k = E.shape[-1]
    output = []
    for _ in range(n):
        i = I[:, _]
        e = E[:, _]
        f = F[:, _]
        s = S[:, _]
        
        m = f * m + torch.einsum("... k, ... d -> ... k d", e, i)
        y = torch.einsum("... k d, ... k -> ... d", m, s)
        output.append(y.unsqueeze(1))
        
    return torch.cat(output, dim=1)