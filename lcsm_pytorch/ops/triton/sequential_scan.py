# import torch
# import triton

# @triton.jit
# def fwd_recurrence(
#     i,
#     e,
#     f,
#     is_e_dependent,
#     is_f_dependent,
#     is_s_dependent,
# ):


# from mnet_pytorch.utils import is_dependent

# class PscanBlock(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, i, e, f, s):
#         # i: b, n, d
#         # e: b, n, k or k
#         # f: b, n, k, d or k, d
#         # s: b, n, k or k
#         is_e_dependent = is_dependent(e)
#         is_f_dependent = is_dependent(f)
#         is_s_dependent = is_dependent(s)

#         b, n, d = i.shape
#         k = e.shape[-1]

#         grid =

#     @staticmethod
#     def backward(ctx, do):
#         i, e, f, s = ctx.saved_tensors
#         C = ctx.C
