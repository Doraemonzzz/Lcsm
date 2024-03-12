import torch
import torch.nn.functional as F

from mnet_pytorch.utils import complex_log, is_dependent


def pscan(i, e, f, s, m0=None):
    # i: b, n, d
    # e: b, n, k or k
    # f: b, n, k, d or k, d
    # s: b, n, k or k
    b, n, d = i.shape
    # construct memory
    m_bar = torch.einsum("... k, ... d -> ... k d", e, i)  # b, n, k, d
    if m0 == None:
        b, n, k, d = m_bar.shape
        m0 = torch.zeros(b, 1, k, d).to(m_bar)
    m_bar = torch.cat([m0, m_bar], dim=-3)

    log_f = complex_log(f)
    log_m_bar = complex_log(m_bar)
    if is_dependent(f):  # data dependent
        f_star = F.pad(torch.cumsum(log_f, dim=-3), (0, 0, 0, 0, 1, 0))
    else:  # data independent
        f_star = torch.arange(n + 1).reshape(1, -1, 1, 1).to(log_f) * log_f

    log_m0_plus_m_star = torch.logcumsumexp(log_m_bar - f_star, dim=-3)
    log_m = f_star + log_m0_plus_m_star
    m = torch.exp(log_m).real[:, 1:].to(i.dtype)

    o = torch.einsum("... k d, ... k -> ... d", m, s)

    return o
