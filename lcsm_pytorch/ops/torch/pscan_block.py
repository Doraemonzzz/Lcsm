import torch
import torch.nn.functional as F

from lcsm_pytorch.utils import complex_log, is_dependent


##### Block version
def pscan_fn_fwd(
    i, e, f, m0, s_arry, dims=(-2), reverse=False, f0=None, return_last=True
):
    # i: b, n, d
    # e: b, n, k or k
    # f: b, n, k, d or k, d
    # s: b, n, k or k

    b, n, d = i.shape
    # construct memory
    m_bar = torch.einsum("... k, ... d -> ... k d", e, i)  # b, n, k, d
    m_bar = torch.cat([m0, m_bar], dim=-3)

    if is_dependent(f) and reverse:
        f = torch.cat([f0, f], dim=-3)[:, :-1]

    log_f = complex_log(f)
    log_m_bar = complex_log(m_bar)
    if is_dependent(f):  # data dependent
        f_star = F.pad(torch.cumsum(log_f, dim=-3), (0, 0, 0, 0, 1, 0))
    else:  # data independent
        f_star = torch.arange(n + 1).reshape(1, -1, 1, 1).to(log_f) * log_f

    log_m0_plus_m_star = torch.logcumsumexp(log_m_bar - f_star, dim=-3)
    log_m = f_star + log_m0_plus_m_star

    m = torch.exp(log_m).real.to(i.dtype)

    output = []
    for j, dim in enumerate(dims):
        if dim == -2:
            pattern = "... k d, ... k -> ... d"
        else:
            pattern = "... k d, ... d -> ... k"

        output.append(torch.einsum(pattern, m[:, 1:], s_arry[j]))

    if return_last:
        return output, m[:, -1:]
    else:
        return output, m


def pscan_fn_bwd(i, e, f, m0, m_, s_arry, dims=(-2), reverse=False, f0=None):
    # i: b, n, d
    # e: b, n, k or k
    # f: b, n, k, d or k, d
    # s: b, n, k or k

    b, n, d = i.shape
    # construct memory
    m_bar = torch.einsum("... k, ... d -> ... k d", e, i)  # b, n, k, d
    m_bar = torch.cat([m0, m_bar], dim=-3)

    if is_dependent(f) and reverse:
        f = torch.cat([f0, f], dim=-3)[:, :-1]

    log_f = complex_log(f)
    log_m_bar = complex_log(m_bar)
    if is_dependent(f):  # data dependent
        f_star = F.pad(torch.cumsum(log_f, dim=-3), (0, 0, 0, 0, 1, 0))
    else:  # data independent
        f_star = torch.arange(n + 1).reshape(1, -1, 1, 1).to(log_f) * log_f

    log_m0_plus_m_star = torch.logcumsumexp(log_m_bar - f_star, dim=-3)
    log_m = f_star + log_m0_plus_m_star

    m = torch.exp(log_m).real.to(i.dtype)

    df = m[:, 1:] * m_

    output = []
    for j, dim in enumerate(dims):
        if dim == -2:
            pattern = "... k d, ... k -> ... d"
        else:
            pattern = "... k d, ... d -> ... k"

        output.append(torch.einsum(pattern, m[:, 1:], s_arry[j]))

    return output, m[:, -1:], df


class PscanBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, e, f, s, C=128):
        # i: b, n, d
        # e: b, n, k or k
        # f: b, n, k, d or k, d
        # s: b, n, k or k
        b, n, d = i.shape
        k = e.shape[-1]
        is_e_dependent = is_dependent(e)
        is_f_dependent = is_dependent(f)
        is_s_dependent = is_dependent(s)

        T = (n + C - 1) // C
        m = torch.zeros(b, 1, k, d).to(i)
        o = []

        for t in range(T):
            start = t * C
            end = min(n, start + C)
            # get chunk
            it = i[:, start:end]

            if is_e_dependent:
                et = e[:, start:end]
            else:
                et = e

            if is_f_dependent:
                ft = f[:, start:end]
            else:
                ft = f

            if is_s_dependent:
                st = s[:, start:end]
            else:
                st = s

            output, m = pscan_fn_fwd(it, et, ft, m, (st,), dims=(-2,))
            ot = output[0]
            o.append(ot)

        o = torch.cat(o, dim=-2)

        ctx.save_for_backward(i, e, f, s)
        ctx.C = C

        return o

    @staticmethod
    def backward(ctx, do):
        i, e, f, s = ctx.saved_tensors
        C = ctx.C

        ##### ds
        b, n, d = i.shape
        k = e.shape[-1]
        is_e_dependent = is_dependent(e)
        is_f_dependent = is_dependent(f)
        is_s_dependent = is_dependent(s)

        T = (n + C - 1) // C
        m = torch.zeros(b, 1, k, d).to(i)
        ds = []
        M = []

        for t in range(T):
            start = t * C
            end = min(n, start + C)
            # get chunk
            it = i[:, start:end]
            dot = do[:, start:end]

            if is_e_dependent:
                et = e[:, start:end]
            else:
                et = e

            if is_f_dependent:
                ft = f[:, start:end]
            else:
                ft = f

            if is_s_dependent:
                st = s[:, start:end]
            else:
                st = s

            output, m_array = pscan_fn_fwd(
                it, et, ft, m, (dot,), dims=(-1,), return_last=False
            )
            m = m_array[:, -1:]
            M.append(m_array[:, :-1])
            dst = output[0]
            ds.append(dst)

        M = torch.cat(M, dim=1)
        ds = torch.cat(ds, dim=-2)

        ##### di, de, df
        di = []
        de = []
        df = []

        do_flip = torch.flip(do, dims=[-2])
        if is_f_dependent:
            f0 = torch.zeros(b, 1, k, d).to(f)
        else:
            f0 = torch.zeros_like(f).to(f)
        # do_last = do[:, -1:]
        i_flip = torch.flip(i, dims=[-2])

        if is_s_dependent:
            s_flip = torch.flip(s, dims=[-2])
        else:
            s_flip = s

        if is_e_dependent:
            e_flip = torch.flip(e, dims=[-2])

        else:
            e_flip = e

        if is_f_dependent:
            f_flip = torch.flip(f, dims=[-3])
        else:
            f_flip = f

        M_flip = torch.flip(M, dims=[1])

        dm = torch.zeros(b, 1, k, d).to(i)
        de = []
        di = []
        df = []

        for t in range(T):
            # print(dm[0, 0, :3, :3])
            start = t * C
            end = min(n, start + C)
            # get chunk
            dot = do_flip[:, start:end]
            it = i_flip[:, start:end]
            mt = M_flip[:, start:end]

            if is_s_dependent:
                st = s_flip[:, start:end]
            else:
                st = s_flip

            if is_e_dependent:
                et = e_flip[:, start:end]
            else:
                et = e_flip

            if is_f_dependent:
                ft = f_flip[:, start:end]
            else:
                ft = f_flip

            output, dm, dft = pscan_fn_bwd(
                dot, st, ft, dm, mt, (et, it), dims=(-2, -1), reverse=True, f0=f0
            )
            dit = output[0]
            det = output[1]

            if is_f_dependent:
                f0 = ft[:, -1:]

            di.append(dit)
            de.append(det)
            df.append(dft)

        di = torch.flip(torch.cat(di, dim=-2), dims=[-2])
        de = torch.flip(torch.cat(de, dim=-2), dims=[-2])
        df = torch.flip(torch.cat(df, dim=1), dims=[1])

        if not is_s_dependent:
            ds = ds.sum(dim=0).sum(dim=0)
        if not is_e_dependent:
            de = de.sum(dim=0).sum(dim=0)
        if not is_f_dependent:
            df = df.sum(dim=0).sum(dim=0)

        return di, de, df, ds, None


pscan_block = PscanBlock.apply
