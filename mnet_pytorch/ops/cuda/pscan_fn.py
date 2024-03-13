import torch
from einops import rearrange, repeat
from torch.autograd import Function

import pscan_cuda
from mnet_pytorch.utils import is_dependent


class ScanCuda(Function):
    @staticmethod
    def forward(ctx, i, e, f, s):
        # i: b, n, d
        # e: b, n, k or k
        # f: b, n, k, d or k, d
        # s: b, n, k or k

        # i = i.transpose(0, 1)
        # if is_dependent(e):
        #     e = e.transpose(0, 1)
        # if is_dependent(f):
        #     f = f.transpose(0, 1)
        # if is_dependent(s):
        #     s = s.transpose(0, 1)

        i = i.contiguous()
        e = e.contiguous()
        f = f.contiguous()
        s = s.contiguous()

        b, n, d = i.shape
        k = e.shape[-1]

        input = torch.einsum("... k, ... d -> ... k d", e, i)

        if not is_dependent(f):
            decay = repeat(f, "k d -> b k d", b=b)
        else:
            decay = f

        input, decay = map(
            lambda x: rearrange(x, "... k d -> ... (k d)"), [input, decay]
        )
        input = input.contiguous()
        decay = decay.contiguous()
        print(input.shape, decay.shape)
        m = pscan_cuda.forward(input, decay)

        m = rearrange(m, "... (k d) -> ... k d", k=k).contiguous()
        output = torch.einsum("... k d, ... k -> ... d", m, s)

        ctx.save_for_backward(i, e, f, s, m)

        return output  # .transpose(0, 1)

    @staticmethod
    def backward(ctx, do):
        do = do.contiguous()
        i, e, f, s, m = ctx.saved_tensors

        b, n, d = i.shape
        k = e.shape[-1]

        input = torch.einsum("... k, ... d -> ... k d", e, i)
        grad = torch.einsum("... k, ... d -> ... k d", s, do)

        if not is_dependent(f):
            decay = repeat(f, "k d -> b k d", b=b)
        else:
            decay = f

        input, decay = map(
            lambda x: rearrange(x, "... k d -> ... (k d)"), [input, decay]
        )
        input = input.contiguous()
        decay = decay.contiguous()

        dm, df = pscan_cuda.backward(input, decay, m, grad)

        dm, df = map(lambda x: rearrange(x, "... (k d) -> ... k d", k=k), [dm, df])
        print(m.shape, do.shape)
        ds = torch.einsum("... k d, ... d -> ... k", m, do)
        de = torch.einsum("... k d, ... d -> ... k", dm, i)
        di = torch.einsum("... k d, ... k -> ... d", dm, e)  # .transpose(0, 1)

        if not is_dependent(f):
            df = df.sum(0).sum(0)
        # else:
        #     df = df.transpose(0, 1)

        if not is_dependent(s):
            ds = ds.sum(0).sum(0)
        # else:
        #     ds = ds.transpose(0, 1)

        if not is_dependent(e):
            de = de.sum(0).sum(0)
        # else:
        #     de = de.transpose(0, 1)

        return di, de, df, ds


pscan_cuda_fn = ScanCuda.apply
