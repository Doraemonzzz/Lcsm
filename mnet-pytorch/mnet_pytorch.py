from einops import rearrange, repeat
import torch
import torch.nn.functional as F
import pytest
import os
import triton

def is_dependent(x):
    return len(x.shape) >= 3

def process(x, i):
    if is_dependent(x):
        return x[:, i]
    else:
        return x

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

##### parallel version
def complex_log(x, eps=1e-6):
    eps = x.new_tensor(eps)
    real = x.abs().maximum(eps).log()
    imag = (x < 0).to(x.dtype) * torch.pi
    return torch.complex(real.to(torch.float), imag.to(torch.float))

def pscan(i, e, f, s, m0=None):
    # i: b, n, d
    # e: b, n, k or k
    # f: b, n, k, d or k, d
    # s: b, n, k or k
    b, n, d = i.shape
    # construct memory
    m_bar = torch.einsum("... k, ... d -> ... k d", e, i) # b, n, k, d
    if m0 == None:
        b, n, k, d = m_bar.shape
        m0 = torch.zeros(b, 1, k, d).to(m_bar)
    m_bar = torch.cat([m0, m_bar], dim=-3)
    
    log_f = complex_log(f)
    log_m_bar = complex_log(m_bar)
    if is_dependent(f): # data dependent
        f_star = F.pad(torch.cumsum(log_f, dim=-3), (0, 0, 0, 0, 1, 0))      
    else: # data independent
        f_star = torch.arange(n + 1).reshape(1, -1, 1, 1).to(log_f) * log_f

    log_m0_plus_m_star = torch.logcumsumexp(log_m_bar - f_star, dim=-3)
    log_m = f_star + log_m0_plus_m_star
    m = torch.exp(log_m).real[:, 1:].to(i.dtype)

    o = torch.einsum("... k d, ... k -> ... d", m, s)
    
    return o

class Pscan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, e, f, s, m0=None):
        # i: b, n, d
        # e: b, n, k or k
        # f: b, n, k, d or k, d
        # s: b, n, k or k
        b, n, d = i.shape
        # construct memory
        m_bar = torch.einsum("... k, ... d -> ... k d", e, i) # b, n, k, d
        if m0 == None:
            b, n, k, d = m_bar.shape
            m0 = torch.zeros(b, 1, k, d).to(m_bar)
        m_bar = torch.cat([m0, m_bar], dim=-3)
        
        log_f = complex_log(f)
        log_m_bar = complex_log(m_bar)
        if is_dependent(f): # data dependent
            f_star = F.pad(torch.cumsum(log_f, dim=-3), (0, 0, 0, 0, 1, 0))      
        else: # data independent
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
        m_bar = torch.einsum("... k, ... d -> ... k d", e, i) # b, n, k, d
        if m0 == None:
            b, n, k, d = m_bar.shape
            m0 = torch.zeros(b, 1, k, d).to(m_bar)
        m_bar = torch.cat([m0, m_bar], dim=-3)
        
        log_f = complex_log(f)
        log_m_bar = complex_log(m_bar)
        if is_dependent(f): # data dependent
            f_star = F.pad(torch.cumsum(log_f, dim=-3), (0, 0, 0, 0, 1, 0))      
        else: # data independent
            f_star = torch.arange(n + 1).reshape(1, -1, 1, 1).to(log_f) * log_f

        log_m0_plus_m_star = torch.logcumsumexp(log_m_bar - f_star, dim=-3)
        log_m = f_star + log_m0_plus_m_star
        m = torch.exp(log_m).real.to(i.dtype)
        
        ds = torch.einsum("... k d, ... d -> ... k", m[:, 1:], do)
        
        # others
        dm_bar = torch.einsum("... k, ... d -> ... k d", s, do) # b, n, k, d
        dm_bar = torch.flip(dm_bar, dims=[-3])
        b, n, k, d = dm_bar.shape
        dmn = torch.zeros(b, 1, k, d).to(dm_bar)
        dm_bar = torch.cat([dm_bar, dmn], dim=-3)
        
        if is_dependent(f):
            log_f_reverse = torch.flip(log_f, dims=[-3])
        else:
            log_f_reverse = log_f
        log_dm_bar = complex_log(dm_bar)
        if len(f.shape) > 2: # data dependent
            f_star_reverse = F.pad(torch.cumsum(log_f_reverse, dim=-3), (0, 0, 0, 0, 1, 0))      
        else: # data independent
            f_star_reverse = torch.arange(n + 1).reshape(1, -1, 1, 1).to(log_f) * log_f_reverse

        log_dm0_plus_dm_star = torch.logcumsumexp(log_dm_bar - f_star_reverse, dim=-3)
        log_dm_reverse = f_star_reverse + log_dm0_plus_dm_star
        dm = torch.exp(torch.flip(log_dm_reverse, dims=[-3])).real[:, 1:].to(i.dtype)

        df = dm * m[:, :-1]
        de = torch.einsum("... k d, ... d -> ... k", dm, i)
        di = torch.einsum("... k d, ... k -> ... d", dm, e)
        
        return di, de, df, ds, None
    
pscan_torch = Pscan.apply

@pytest.mark.parametrize('b, n, k, d', 
    [
        (6, 512, 32, 128),
    ]
)
@pytest.mark.parametrize('e_dependent, f_dependent, s_dependent', 
    [
        # (True, True, True),
        (True, False, True),
        # (True, True, False),
        # (False, True, True),
    ]
)
# @pytest.mark.parametrize('dtype', [torch.float32])
@pytest.mark.parametrize('dtype', [torch.bfloat16])
def test_op(b, n, k, d, e_dependent, f_dependent, s_dependent, dtype, device="cuda:0"):
    torch.manual_seed(20)
    i = torch.empty((b, n, d), dtype=dtype, device=device).normal_(mean=0., std=0.5).requires_grad_()
    if e_dependent:
        e = torch.empty((b, n, k), dtype=dtype, device=device).normal_(mean=0., std=0.5).requires_grad_()
    else:
        e = torch.empty((k), dtype=dtype, device=device).normal_(mean=0., std=0.5).requires_grad_()
        
    if f_dependent:
        f = F.sigmoid(torch.empty((b, n, k, d), dtype=dtype, device=device).normal_(mean=0., std=0.5)).requires_grad_()
    else:
        f = F.sigmoid(torch.empty((k, d), dtype=dtype, device=device).normal_(mean=0., std=0.5)).requires_grad_()
        # f = torch.ones((k, d), dtype=dtype, device=device).requires_grad_()
        # f = F.sigmoid(torch.empty((1, 1, k, d), dtype=dtype, device=device).normal_(mean=0., std=0.5)).requires_grad_()
        # f = repeat(f, "x y k d -> (x b) (y n) k d", b=b, n=n)
    
    if s_dependent:
        s = torch.empty((b, n, k), dtype=dtype, device=device).normal_(mean=0., std=0.5).requires_grad_()
    else:
        s = torch.empty((k), dtype=dtype, device=device).normal_(mean=0., std=0.5).requires_grad_()
    
    dout = torch.randn(b, n, d).to(i.device)

    # reference
    ref_out = expand_and_shrink(i, e, f, s)
    ref_out.backward(dout, retain_graph=True)
    ref_di, i.grad = i.grad.clone(), None
    ref_de, e.grad = e.grad.clone(), None
    ref_df, f.grad = f.grad.clone(), None
    ref_ds, s.grad = s.grad.clone(), None
    
    # pscan
    pscan_out = pscan(i, e, f, s)
    pscan_out.backward(dout, retain_graph=True)
    pscan_di, i.grad = i.grad.clone(), None
    pscan_de, e.grad = e.grad.clone(), None
    pscan_df, f.grad = f.grad.clone(), None
    pscan_ds, s.grad = s.grad.clone(), None
    
    # pscan torch
    pscan_torch_out = pscan_torch(i, e, f, s)
    pscan_torch_out.backward(dout, retain_graph=True)
    pscan_torch_di, i.grad = i.grad.clone(), None
    pscan_torch_de, e.grad = e.grad.clone(), None
    pscan_torch_df, f.grad = f.grad.clone(), None
    pscan_torch_ds, s.grad = s.grad.clone(), None

    print("naive Vs pscan")
    print(torch.norm(ref_out.float() - pscan_out.float()))
    print(torch.norm(ref_di.float() - pscan_di.float()))
    print(torch.norm(ref_de.float() - pscan_de.float()))
    print(torch.norm(ref_df.float() - pscan_df.float()))
    print(torch.norm(ref_ds.float() - pscan_ds.float()))
    print("naive Vs pscan torch")
    print(torch.norm(ref_out.float() - pscan_torch_out.float()))
    print(torch.norm(ref_di.float() - pscan_torch_di.float()))
    print(torch.norm(ref_de.float() - pscan_torch_de.float()))
    print(torch.norm(ref_df.float() - pscan_torch_df.float()))
    print(torch.norm(ref_ds.float() - pscan_torch_ds.float()))
    
    
    torch.testing.assert_close(ref_out.float(), pscan_out.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ref_di.float(), pscan_di.float(), atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(ref_de.float(), pscan_de.float(), atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(ref_df.float(), pscan_df.float(), atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(ref_ds.float(), pscan_ds.float(), atol=5e-2, rtol=1e-2)


# b, n, k, d = 6, 512, 32, 128
# device = "cuda:0"
# speed_configs = [triton.testing.Benchmark(
#     x_names=['N_CTX'],
#     # x_vals=[2**i for i in range(10, 16)],
#     # x_vals=[2**i for i in range(10, 15)],
#     x_vals=[2**i for i in range(10, 14)],
#     # x_vals=[2**i for i in range(10, 13)],
#     line_arg='provider',
#     # line_vals=['triton', "linear", "softmax", "xformers"] + (['flash'] if HAS_FLASH else []),
#     # line_names=['Triton', "Linear", "Softmax", "Xformers"] + (['Flash'] if HAS_FLASH else []),
#     line_vals=['attention_lrpe', 'lightning_lrpe'] + (['flash'] if HAS_FLASH else []),
#     line_names=['attention_lrpe', 'lightning_lrpe'] + (['Flash'] if HAS_FLASH else []),
#     # line_vals=['triton',] + (['xformers']),
#     # line_names=['Triton',] + (['Xformers']),
#     # line_vals=['triton', "xformers"] + (['flash'] if HAS_FLASH else []),
#     # line_names=['Triton', "Xformers"] + (['Flash'] if HAS_FLASH else []),
#     # line_vals=["xformers"],
#     # line_names=["Xformers"],
#     styles=[('red', '-'), ('orange', '-'), ('green', '-'), ('blue', '-'), ('black', '-')],
#     ylabel='ms',
#     plot_name=f'fused-linear-attention-batch{BATCH}-head{N_HEADS}-qk{QK_HEAD}-v{V_HEAD}-{mode}-causal-{causal}',
#     args={'H': N_HEADS, 'BATCH': BATCH, 'QK_HEAD': QK_HEAD, 'V_HEAD': V_HEAD, 'dtype': torch.float16, 'mode': mode, 'causal': causal}
# ) for mode in ['fwd', 'bwd'] for causal in [True]]


# @triton.testing.perf_report(speed_configs)
# def bench_flash_attention_speed(b, n, k, d, mode, provider, dtype=torch.float16, device="cuda"):
#     assert mode in ['fwd', 'bwd']
#     warmup = 25
#     rep = 100
#     if provider == "attention_lrpe":
#         q = torch.randn((BATCH, H, N_CTX, QK_HEAD), dtype=dtype, device=device, requires_grad=True)
#         k = torch.randn((BATCH, H, N_CTX, QK_HEAD), dtype=dtype, device=device, requires_grad=True)
#         v = torch.randn((BATCH, H, N_CTX, V_HEAD), dtype=dtype, device=device, requires_grad=True)
#         theta = torch.randn((H, QK_HEAD), dtype=dtype, device="cuda")
#         # slopes = torch.empty(0).to(q)
#         # fn = lambda: attention(q, k, v, causal, slopes)
#         fn = lambda: attention_lrpe(q, k, v, theta)
#         if mode == 'bwd':
#             o = fn()
#             do = torch.randn_like(o)
#             fn = lambda: o.backward(do, retain_graph=True)
#     if provider == "lightning_lrpe":
#         q = torch.randn((BATCH, H, N_CTX, QK_HEAD), dtype=dtype, device=device, requires_grad=True)
#         k = torch.randn((BATCH, H, N_CTX, QK_HEAD), dtype=dtype, device=device, requires_grad=True)
#         v = torch.randn((BATCH, H, N_CTX, V_HEAD), dtype=dtype, device=device, requires_grad=True)
#         theta = torch.randn((H, QK_HEAD), dtype=dtype, device="cuda")
#         # slopes = torch.empty(0).to(q)
#         # fn = lambda: attention(q, k, v, causal, slopes)
#         fn = lambda: lightning_lrpe(q, k, v, theta)
#         if mode == 'bwd':
#             o = fn()
#             do = torch.randn_like(o)
#             fn = lambda: o.backward(do, retain_graph=True)
#     elif provider == "flash":
#         q = torch.randn((BATCH, N_CTX, H, QK_HEAD), dtype=dtype, device=device, requires_grad=True)
#         k = torch.randn((BATCH, N_CTX, H, QK_HEAD), dtype=dtype, device=device, requires_grad=True)
#         v = torch.randn((BATCH, N_CTX, H, V_HEAD), dtype=dtype, device=device, requires_grad=True)
#         sm_scale = 1.3
#         fn = lambda: flash_wrapper(q, k, v, 0., sm_scale, causal)
#         if mode == 'bwd':
#             o = fn()
#             do = torch.randn_like(o)
#             fn = lambda: o.backward(do, retain_graph=True)
#     elif provider in ["linear", "softmax"]:
#         q = torch.randn((BATCH, H, N_CTX, QK_HEAD), dtype=dtype, device=device, requires_grad=True)
#         k = torch.randn((BATCH, H, N_CTX, QK_HEAD), dtype=dtype, device=device, requires_grad=True)
#         v = torch.randn((BATCH, H, N_CTX, V_HEAD), dtype=dtype, device=device, requires_grad=True)
#         slopes = _build_csb_tensor(H).to(q)
#         M = get_full_mask(N_CTX, slopes).to(q)
#         fn = lambda: naive_linear(q, k, v, causal, M)
#         if mode == 'bwd':
#             o = fn()
#             do = torch.randn_like(o)
#             fn = lambda: o.backward(do, retain_graph=True)
#     elif provider == "xformers":
#         q = torch.randn((BATCH, N_CTX, H, QK_HEAD), dtype=dtype, device=device, requires_grad=True)
#         k = torch.randn((BATCH, N_CTX, H, QK_HEAD), dtype=dtype, device=device, requires_grad=True)
#         v = torch.randn((BATCH, N_CTX, H, V_HEAD), dtype=dtype, device=device, requires_grad=True)
#         # if causal:
#         #     fn = lambda: xops.memory_efficient_attention(q, k, v, attn_bias=xops.LowerTriangularMask())
#         # else:
#         #     fn = lambda: xops.memory_efficient_attention(q, k, v)
#         fn = lambda: xformer_wrapper(q, k, v, causal)
#         if mode == 'bwd':
#             o = fn()
#             do = torch.randn_like(o)
#             fn = lambda: o.backward(do, retain_graph=True)
#     ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

#     return ms


    
save_path = 'stat' 
os.makedirs(save_path, exist_ok=True)
# only works on post-Ampere GPUs right now
# bench_flash_attention_speed.run(save_path=save_path, print_data=True)
# bench_flash_attention_memory.run(save_path=save_path, print_data=True)
# test_op(save_path='.', print_data=True)