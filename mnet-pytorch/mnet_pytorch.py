from einops import rearrange, repeat
import torch
import torch.nn.functional as F
import pytest
import os
import triton
import numpy as np

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

##### Block version
def pscan_fn(i, e, f, m0, s_arry, dims=(-2)):
    # i: b, n, d
    # e: b, n, k or k
    # f: b, n, k, d or k, d
    # s: b, n, k or k
    b, n, d = i.shape
    # construct memory
    m_bar = torch.einsum("... k, ... d -> ... k d", e, i) # b, n, k, d
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

    output = []
    for i, dim in enumerate(dims):
        if dim == -2:
            pattern = "... k d, ... k -> ... d"
        else:
            pattern = "... k d, ... d -> ... k"
        output.append(torch.einsum(pattern, m, s_arry[i]))
    
    return output, m[:, -1:]

class PscanBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, e, f, s, C=256):
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
                
            output, m = pscan_fn(it, et, ft, m, (st,), dims=(-2,))
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
                
            output, m = pscan_fn(it, et, ft, m, (dot,), dims=(-1,))
            dst = output[0]
            ds.append(dst)
        
        ds = torch.cat(ds, dim=-2)
        
        ##### di, de, df
        di = []
        de = []
        df = []
        
        do_flip = torch.flip(do, dims=[-2])
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
            f_flip = torch.flip(f, dims=[-2])
        else:
            f_flip = f
            
        m = torch.zeros(b, 1, k, d).to(i)
        de = []
        di = []
        
        for t in range(T):
            start = t * C
            end = min(n, start + C)
            # get chunk
            dot = do_flip[:, start:end]
            it = i_flip[:, start:end]
            
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
                
            output, m = pscan_fn(dot, st, ft, m, (et, it), dims=(-2, -1))
            dit = output[0]
            det = output[1]
            
            de.append(det)
            di.append(dit)
            
        de = torch.flip(torch.cat(de, dim=-2), dims=[-2])
        di = torch.flip(torch.cat(di, dim=-2), dims=[-2])
        
        # de = torch.cat(de, dim=-2)
        # di = torch.cat(di, dim=-2)

        df = None
        
        return di, de, df, ds, None

pscan_block = PscanBlock.apply

@pytest.mark.parametrize('b, n, k, d', 
    [
        # (6, 512, 32, 128),
        (6, 2048, 32, 128),
    ]
)
@pytest.mark.parametrize('e_dependent, f_dependent, s_dependent', 
    [
        (True, True, True),
        # (True, True, False),
        # (True, False, True),
        # (True, False, False),
        # (False, True, True),
        # (False, True, False),
        # (False, False, True),
        # (False, False, False),
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
    
    # pscan block
    pscan_block_out = pscan_block(i, e, f, s)
    pscan_block_out.backward(dout, retain_graph=True)
    pscan_block_di, i.grad = i.grad.clone(), None
    pscan_block_de, e.grad = e.grad.clone(), None
    # pscan_block_df, f.grad = f.grad.clone(), None
    pscan_block_ds, s.grad = s.grad.clone(), None

    print("naive Vs pscan")
    print(f"out: {torch.norm(ref_out.float() - pscan_out.float())}")
    print(f"di: {torch.norm(ref_di.float() - pscan_di.float())}")
    print(f"de: {torch.norm(ref_de.float() - pscan_de.float())}")
    print(f"df: {torch.norm(ref_df.float() - pscan_df.float())}")
    print(f"ds: {torch.norm(ref_ds.float() - pscan_ds.float())}")
    print("naive Vs pscan torch")
    print(f"out: {torch.norm(ref_out.float() - pscan_torch_out.float())}")
    print(f"di: {torch.norm(ref_di.float() - pscan_torch_di.float())}")
    print(f"de: {torch.norm(ref_de.float() - pscan_torch_de.float())}")
    print(f"df: {torch.norm(ref_df.float() - pscan_torch_df.float())}")
    print(f"ds: {torch.norm(ref_ds.float() - pscan_torch_ds.float())}")
    print("naive Vs pscan block")
    print(f"out: {torch.norm(ref_out.float() - pscan_block_out.float())}")
    print(f"di: {torch.norm(ref_di.float() - pscan_block_di.float())}")
    print(f"de: {torch.norm(ref_de.float() - pscan_block_de.float())}")
    # print(f"df: {torch.norm(ref_df.float() - pscan_block_df.float())}")
    print(f"ds: {torch.norm(ref_ds.float() - pscan_block_ds.float())}")
    
    
    torch.testing.assert_close(ref_out.float(), pscan_out.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ref_di.float(), pscan_di.float(), atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(ref_de.float(), pscan_de.float(), atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(ref_df.float(), pscan_df.float(), atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(ref_ds.float(), pscan_ds.float(), atol=5e-2, rtol=1e-2)


b, n, k, d = 6, 512, 32, 128
device = "cuda:0"
speed_configs = [triton.testing.Benchmark(
    x_names=['n'],
    x_vals=[2**i for i in range(9, 13)],
    line_arg='provider',
    line_vals=["origin", "pscan", "pscan_torch", "pscan_block"],
    line_names=["origin", "pscan", "pscan_torch", "pscan_block"],
    styles=[('red', '-'), ('orange', '-'), ('green', '-'), ('blue', '-'), ('black', '-')],
    ylabel='ms',
    plot_name=f'mnet-batch{b}-n{n}-k{k}-d{d}-{mode}',
    args={'b': b, 'k': k, 'd': d, 'dtype': torch.float16, 'mode': mode,}
) for mode in ['fwd', 'bwd']]
@triton.testing.perf_report(speed_configs)
def bench_mnet_speed(b, n, k, d, mode, provider, dtype=torch.bfloat16, device="cuda"):
    assert mode in ['fwd', 'bwd']
    warmup = 25
    rep = 100
    i = torch.empty((b, n, d), dtype=dtype, device=device).normal_(mean=0., std=0.5).requires_grad_()
    e = torch.empty((b, n, k), dtype=dtype, device=device).normal_(mean=0., std=0.5).requires_grad_()
    f = F.sigmoid(torch.empty((b, n, k, d), dtype=dtype, device=device).normal_(mean=0., std=0.5)).requires_grad_()
    s = torch.empty((b, n, k), dtype=dtype, device=device).normal_(mean=0., std=0.5).requires_grad_()
    if provider == "origin":
        fun = expand_and_shrink
    elif provider == "pscan":
        fun = pscan
    elif provider == "pscan_torch":
        fun = pscan_torch
    elif provider == "pscan_block":
        fun = pscan_block
    fn = lambda: fun(i, e, f, s)

    if mode == 'bwd':
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
        
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    return ms

def get_memory(device):
    mb_used = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    torch.cuda.reset_peak_memory_stats(device)

    return mb_used

memory_configs = [triton.testing.Benchmark(
    x_names=['n'],
    x_vals=[2**i for i in range(9, 13)],
    line_arg='provider',
    line_vals=["origin", "pscan", "pscan_torch", "pscan_block"],
    line_names=["origin", "pscan", "pscan_torch", "pscan_block"],
    styles=[('red', '-'), ('orange', '-'), ('green', '-'), ('blue', '-'), ('black', '-')],
    ylabel='ms',
    plot_name=f'mnet-batch{b}-n{n}-k{k}-d{d}-{mode}',
    args={'b': b, 'k': k, 'd': d, 'dtype': torch.float16, 'mode': mode,}
) for mode in ['fwd', 'bwd']]
@triton.testing.perf_report(memory_configs)
def bench_mnet_memory(b, n, k, d, mode, provider, dtype=torch.bfloat16, device="cuda"):
    assert mode in ['fwd', 'bwd']
    warmup = 25
    rep = 5
    i = torch.empty((b, n, d), dtype=dtype, device=device).normal_(mean=0., std=0.5).requires_grad_()
    e = torch.empty((b, n, k), dtype=dtype, device=device).normal_(mean=0., std=0.5).requires_grad_()
    f = F.sigmoid(torch.empty((b, n, k, d), dtype=dtype, device=device).normal_(mean=0., std=0.5)).requires_grad_()
    s = torch.empty((b, n, k), dtype=dtype, device=device).normal_(mean=0., std=0.5).requires_grad_()
    if provider == "origin":
        fun = expand_and_shrink
    elif provider == "pscan":
        fun = pscan
    elif provider == "pscan_torch":
        fun = pscan_torch
    elif provider == "pscan_block":
        fun = pscan_block
    fn = lambda: fun(i, e, f, s)

    if mode == 'bwd':
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
        
    try:
        torch.cuda.reset_peak_memory_stats(device)
        mb_arr = []
        for _ in range(rep):
            fn()
            mb_arr.append(get_memory(device))
        mb = np.mean(mb_arr)
    except:
        mb = -1

    return mb
    
save_path = 'stat' 
os.makedirs(save_path, exist_ok=True)
# only works on post-Ampere GPUs right now
# bench_mnet_speed.run(save_path=save_path, print_data=True)
# bench_mnet_memory.run(save_path=save_path, print_data=True)
# test_op(save_path='.', print_data=True)