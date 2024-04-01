import torch
from einops import rearrange
from torch.autograd import Function

import pscan_cuda
from lcsm_pytorch.utils import is_dependent


class ScanCuda(Function):
    @staticmethod
    def forward(ctx, i, e, o, s):
        # i: b, n, d
        # e: b, n, k or k or k, d
        # o: b, n, k, d or k, d
        # s: b, n, k or k or k, d

        i = i.contiguous()
        e = e.to(i.dtype).contiguous()
        o = o.to(i.dtype).contiguous()
        s = s.to(i.dtype).contiguous()

        b, n, d = i.shape
        if len(e.shape) != 2:
            k = e.shape[-1]
        else:
            k = e.shape[0]

        # input = torch.einsum("... k, ... d -> ... k d", e, i)
        if len(e.shape) != 2:
            # e:
            # b n k -> b n k 1
            # k -> k 1
            # i:
            # b n d -> b n 1 d
            # ... k 1, b n 1 d -> b n k d
            input = e.unsqueeze(-1) * i.unsqueeze(-2)
        else:
            # k d, b n 1 d -> b n k d
            input = e * i.unsqueeze(-2)

        o_state = f

        is_fdd = is_dependent(f)

        input, o_state = map(
            lambda x: rearrange(x, "... k d -> ... (k d)"), [input, o_state]
        )
        input = input.contiguous()
        o_state = o_state.contiguous()

        m = pscan_cuda.forward(input, o_state.to(input.dtype), is_fdd)

        m = rearrange(m, "... (k d) -> ... k d", k=k)

        if len(s.shape) != 2:
            # s:
            # b n k -> b n k 1
            # k -> k 1
            # ... k d, ... k 1 -> ... d
            output = (m * s.unsqueeze(-1)).sum(dim=-2)
        else:
            # ... k d, ... k d -> ... d
            output = (m * s).sum(dim=-2)
        # output = torch.einsum("... k d, ... k -> ... d", m, s)

        ctx.save_for_backward(i, e, o, s, m.to(i.dtype))

        return output

    @staticmethod
    def backward(ctx, dy):
        i, e, o, s, m = ctx.saved_tensors
        dy = dy.contiguous().to(i.dtype)

        b, n, d = i.shape
        if len(e.shape) != 2:
            k = e.shape[-1]
        else:
            k = e.shape[0]

        is_fdd = is_dependent(f)
        learned = f.requires_grad

        # input = torch.einsum("... k, ... d -> ... k d", e, i)
        # grad = torch.einsum("... k, ... d -> ... k d", s, dy)

        if len(e.shape) != 2:
            # e:
            # b n k -> b n k 1
            # k -> k 1
            # i:
            # b n d -> b n 1 d
            # ... k 1, b n 1 d -> b n k d
            input = e.unsqueeze(-1) * i.unsqueeze(-2)
        else:
            # k d, b n 1 d -> b n k d
            input = e * i.unsqueeze(-2)

        if len(s.shape) != 2:
            # e:
            # b n k -> b n k 1
            # k -> k 1
            # i:
            # b n d -> b n 1 d
            # ... k 1, b n 1 d -> b n k d
            grad = s.unsqueeze(-1) * dy.unsqueeze(-2)
        else:
            # k d, b n 1 d -> b n k d
            grad = s * dy.unsqueeze(-2)

        o_state = o

        input, o_state = map(
            lambda x: rearrange(x, "... k d -> ... (k d)"), [input, o_state]
        )
        input = input.contiguous()
        o_state = o_state.contiguous()

        dm, do = pscan_cuda.backward(
            input, o_state.to(input.dtype), m, grad.to(input.dtype), is_fdd, learned
        )

        dm, do = map(lambda x: rearrange(x, "... (k d) -> ... k d", k=k), [dm, do])

        dm = dm.to(i.dtype)
        do = do.to(i.dtype)

        if len(s.shape) != 2:
            ds = torch.einsum("... k d, ... d -> ... k", m, dy)
        else:
            ds = torch.einsum("... k d, ... d -> ... k d", m, dy)

        if len(e.shape) != 2:
            de = torch.einsum("... k d, ... d -> ... k", dm, i)
        else:
            de = torch.einsum("... k d, ... d -> ... k d", dm, i)
        # di = torch.einsum("... k d, ... k -> ... d", dm, e)

        if len(e.shape) != 2:
            # e:
            # b n k -> b n k 1
            # k -> k 1
            # ... k d, ... 1 d -> b n d
            di = (dm * e.unsqueeze(-1)).sum(dim=-2)
        else:
            # ... k d, ... k d -> b n d
            di = (dm * e).sum(dim=-2)

        if not is_dependent(f):
            do = do.sum(0).sum(0)

        if not is_dependent(s):
            ds = ds.sum(0).sum(0)

        if not is_dependent(e):
            de = de.sum(0).sum(0)

        if not learned:
            do = None

        return di, de, do, ds


pscan_cuda_fn = ScanCuda.apply
