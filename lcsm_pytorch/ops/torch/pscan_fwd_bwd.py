import torch
import torch.nn.functional as F

from lcsm_pytorch.utils import complex_log, is_dependent


class Pscan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, e, f, s, m0=None):
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

        ctx.save_for_backward(i, e, f, s, m0)

        return o

    @staticmethod
    def backward(ctx, do):
        i, e, f, s, m0 = ctx.saved_tensors

        # compute m
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
        m = torch.exp(log_m).real.to(i.dtype)

        ds = torch.einsum("... k d, ... d -> ... k", m[:, 1:], do)

        # others
        dm_bar = torch.einsum("... k, ... d -> ... k d", s, do)  # b, n, k, d
        dm_bar = torch.flip(dm_bar, dims=[-3])
        b, n, k, d = dm_bar.shape
        dmn = torch.zeros(b, 1, k, d).to(dm_bar)
        dm_bar = torch.cat([dm_bar, dmn], dim=-3)

        if is_dependent(f):
            log_f_reverse = torch.flip(log_f, dims=[-3])
        else:
            log_f_reverse = log_f
        log_dm_bar = complex_log(dm_bar)
        if len(f.shape) > 2:  # data dependent
            f_star_reverse = F.pad(
                torch.cumsum(log_f_reverse, dim=-3), (0, 0, 0, 0, 1, 0)
            )
        else:  # data independent
            f_star_reverse = (
                torch.arange(n + 1).reshape(1, -1, 1, 1).to(log_f) * log_f_reverse
            )

        log_dm0_plus_dm_star = torch.logcumsumexp(log_dm_bar - f_star_reverse, dim=-3)
        log_dm_reverse = f_star_reverse + log_dm0_plus_dm_star
        # dm = torch.exp(torch.flip(log_dm_reverse, dims=[-3])).real[:, 1:].to(i.dtype)
        dm = torch.flip(torch.exp(log_dm_reverse).real[:, :-1], dims=[-3]).to(i.dtype)

        df = dm * m[:, :-1]
        de = torch.einsum("... k d, ... d -> ... k", dm, i)
        di = torch.einsum("... k d, ... k -> ... d", dm, e)

        return di, de, df, ds, None


pscan_torch = Pscan.apply
