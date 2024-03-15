import torch
from einops import rearrange
from torch.autograd import Function

import pscan_cuda
from mnet_pytorch.utils import is_dependent


class ScanCuda(Function):
    @staticmethod
    def forward(ctx, i, e, f, s):
        # i: b, n, d
        # e: b, n, k or k or k, d
        # f: b, n, k, d or k, d
        # s: b, n, k or k or k, d

        i = i.contiguous()
        e = e.contiguous()
        f = f.contiguous()
        s = s.contiguous()

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

        decay = f

        is_fdd = is_dependent(f)

        input, decay = map(
            lambda x: rearrange(x, "... k d -> ... (k d)"), [input, decay]
        )
        input = input.contiguous()
        decay = decay.contiguous()

        m = pscan_cuda.forward(input, decay.to(input.dtype), is_fdd)

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

        ctx.save_for_backward(i, e, f, s, m)

        return output

    @staticmethod
    def backward(ctx, do):
        do = do.contiguous()
        i, e, f, s, m = ctx.saved_tensors

        b, n, d = i.shape
        if len(e.shape) != 2:
            k = e.shape[-1]
        else:
            k = e.shape[0]

        is_fdd = is_dependent(f)
        learned = f.requires_grad

        # input = torch.einsum("... k, ... d -> ... k d", e, i)
        # grad = torch.einsum("... k, ... d -> ... k d", s, do)

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
            grad = s.unsqueeze(-1) * do.unsqueeze(-2)
        else:
            # k d, b n 1 d -> b n k d
            grad = s * do.unsqueeze(-2)

        decay = f

        input, decay = map(
            lambda x: rearrange(x, "... k d -> ... (k d)"), [input, decay]
        )
        input = input.contiguous()
        decay = decay.contiguous()

        dm, df = pscan_cuda.backward(
            input, decay.to(input.dtype), m, grad, is_fdd, learned
        )

        dm, df = map(lambda x: rearrange(x, "... (k d) -> ... k d", k=k), [dm, df])

        if len(s.shape) != 2:
            ds = torch.einsum("... k d, ... d -> ... k", m, do)
        else:
            ds = torch.einsum("... k d, ... d -> ... k d", m, do)

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
            df = df.sum(0).sum(0)

        if not is_dependent(s):
            ds = ds.sum(0).sum(0)

        if not is_dependent(e):
            de = de.sum(0).sum(0)

        if not learned:
            df = None

        return di, de, df, ds


pscan_cuda_fn = ScanCuda.apply
