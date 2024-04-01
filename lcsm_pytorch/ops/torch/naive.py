import torch
from mnet_pytorch.utils import process


def expand_and_shrink(i, e, f, s, m0=None):
    b, n, d = i.shape
    if len(e.shape) != 2:
        k = e.shape[-1]
    else:
        k = e.shape[0]

    if m0 == None:
        m0 = torch.zeros(k, d).to(i)
    m = m0
    output = []
    for _ in range(n):
        i_ = i[:, _]
        e_ = process(e, _)
        f_ = process(f, _)
        s_ = process(s, _)

        # m = f_ * m + torch.einsum("... k, ... d -> ... k d", e_, i_)
        # y = torch.einsum("... k d, ... k -> ... d", m, s_)
        if len(e_.shape) != 2:
            input = e_.unsqueeze(-1) * i_.unsqueeze(-2)
        else:
            input = e_ * i_.unsqueeze(-2)

        m = f_ * m + input

        if len(s_.shape) != 2:
            y = (m * s_.unsqueeze(-1)).sum(dim=-2)
        else:
            # ... k d, ... k d -> ... d
            y = (m * s_).sum(dim=-2)

        output.append(y.unsqueeze(1))

    return torch.cat(output, dim=1)
