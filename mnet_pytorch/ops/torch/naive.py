import torch

from mnet_pytorch.utils import process

def expand_and_shrink(i, e, f, s, m0=None):
    b, n, d = i.shape
    k = e.shape[-1]
    if m0 == None:
        m0 = torch.zeros(k, d).to(i)
    m = m0
    output = []
    for _ in range(n):
        i_ = i[:, _]
        # e_ = e[:, _]
        # f_ = f[:, _]
        # s_ = s[:, _]
        e_ = process(e, _)
        f_ = process(f, _)
        s_ = process(s, _)
        # print(f_.shape, m.shape)
        # assert False
        m = f_ * m + torch.einsum("... k, ... d -> ... k d", e_, i_)
        y = torch.einsum("... k d, ... k -> ... d", m, s_)
        output.append(y.unsqueeze(1))
        
    return torch.cat(output, dim=1)

