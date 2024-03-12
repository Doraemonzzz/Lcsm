import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load
import os
import mnet_cuda


class MnetFunction(Function):
    @staticmethod
    def forward(ctx, i, e, f, s):
        # i: b, n, d
        # e: b, n, k or k
        # f: b, n, k, d or k, d
        # s: b, n, k or k
        i = i.contiguous()
        e = e.contiguous()
        f = f.contiguous()
        s = s.contiguous()
        
        b, n, d = i.shape
        k = e.shape[-1]
        m = torch.zeros(b, 1, k, d).to(i)
        
        output = mnet_cuda.forward(i, e, f, s, m)
        
        ctx.save_for_backward(i, e, f, s)

        return output

    @staticmethod
    def backward(ctx, do):
        do = do.contiguous()
        i, e, f, s = ctx.saved_tensors
        
        b, n, d = i.shape
        k = e.shape[-1]
        m = torch.zeros(b, 1, k, d).to(i)
        
        di, de, df, ds = mnet_cuda.backward(i, e, f, s, m)

        return di, de, df, ds
