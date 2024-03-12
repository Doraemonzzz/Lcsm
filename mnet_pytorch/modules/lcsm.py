import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat

def complex_log(x, eps=1e-6):
    eps = x.new_tensor(eps)
    real = x.abs().maximum(eps).log()
    imag = (x < 0).to(x.dtype) * torch.pi
    return torch.complex(real.to(torch.float), imag.to(torch.float))

def is_dependent(x):
    return len(x.shape) >= 3

##### Block version
def pscan_fn_fwd(i, e, f, m0, s_arry, dims=(-2), reverse=False, f0=None, return_last=True):
    # i: b, n, d
    # e: b, n, k or k
    # f: b, n, k, d or k, d
    # s: b, n, k or k
    
    b, n, d = i.shape
    # construct memory
    m_bar = torch.einsum("... k, ... d -> ... k d", e, i) # b, n, k, d
    m_bar = torch.cat([m0, m_bar], dim=-3) 
    
    if is_dependent(f) and reverse:
        f = torch.cat([f0, f], dim=-3)[:, :-1]
    
    log_f = complex_log(f)
    log_m_bar = complex_log(m_bar)
    if is_dependent(f): # data dependent
        f_star = F.pad(torch.cumsum(log_f, dim=-3), (0, 0, 0, 0, 1, 0))
    else: # data independent
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
    m_bar = torch.einsum("... k, ... d -> ... k d", e, i) # b, n, k, d
    m_bar = torch.cat([m0, m_bar], dim=-3) 
    
    if is_dependent(f) and reverse:
        f = torch.cat([f0, f], dim=-3)[:, :-1]
    
    log_f = complex_log(f)
    log_m_bar = complex_log(m_bar)
    if is_dependent(f): # data dependent
        f_star = F.pad(torch.cumsum(log_f, dim=-3), (0, 0, 0, 0, 1, 0))
    else: # data independent
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
                
            output, m_array = pscan_fn_fwd(it, et, ft, m, (dot,), dims=(-1,), return_last=False)
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
            
            output, dm, dft = pscan_fn_bwd(dot, st, ft, dm, mt, (et, it), dims=(-2, -1), reverse=True, f0=f0)
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

class EOS(nn.Module):
    def __init__(
        self, 
        embed_dim=512,
        expand_dim=8,
        bias=False,
        c_type=0,
        e_type=0,
        f_type=0,
        s_type=0,
        f_learned=True,
        ssm_dim=16,
        tau=16,
        **kwargs,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.expand_dim = expand_dim
        
        self.c_type = c_type
        self.e_type = e_type
        self.f_type = f_type
        self.s_type = s_type
        self.f_learned = f_learned
        
        self.i_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.tau = tau
        
        if self.c_type == 1: # linear
            ##### expand
            if self.e_type == 1:
                self.e_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
            else:
                self.e = nn.Parameter(torch.randn(embed_dim, expand_dim) * 0.1, requires_grad=True)
            
            ##### forget
            if self.f_type == 1: # data dependent
                # d, k -> k d
                self.f1_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
                self.f2_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
            elif self.f_type == 2:
                # 1 d
                self.f_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            elif self.f_type == 3:
                # k 1
                self.f_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
            elif self.f_type == 4: # data independent
                if self.f_learned:
                    f = torch.randn(expand_dim, embed_dim) * 0.1
                else:
                    f = self._build_slope_tensor(expand_dim * embed_dim).reshape(expand_dim, -1)
                # k d
                self.f = nn.Parameter(f, requires_grad=f_learned)
            elif self.f_type == 5:
                if self.f_learned:
                    f = torch.randn(expand_dim) * 0.1
                else:
                    f = self._build_slope_tensor(expand_dim)
                # k 1
                self.f = nn.Parameter(f, requires_grad=self.f_learned)
            elif self.f_type == 6:
                if self.f_learned:
                    f = torch.randn(embed_dim) * 0.1
                else:
                    f = self._build_slope_tensor(embed_dim)
                # 1 d
                self.f = nn.Parameter(torch.randn(embed_dim) * 0.1, requires_grad=True)
            elif self.f_type == 7: # data independent + data dependent
                # k, k d -> k d
                self.f_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
                self.f = nn.Parameter(torch.randn(expand_dim, embed_dim) * 0.1, requires_grad=True)
            elif self.f_type == 8:
                # d, k d -> k d
                self.f_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
                self.f = nn.Parameter(torch.randn(expand_dim, embed_dim) * 0.1, requires_grad=True)
            elif self.f_type == 9:
                # d, k -> k d
                self.f_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
                self.f = nn.Parameter(torch.randn(expand_dim) * 0.1, requires_grad=True)
            elif self.f_type == 10:
                # k, d -> k d
                self.f_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
                self.f = nn.Parameter(torch.randn(embed_dim) * 0.1, requires_grad=True)
                
            ##### shrink
            if self.s_type == 1:
                self.s_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
            else:
                self.s = nn.Parameter(torch.randn(embed_dim, expand_dim) * 0.1, requires_grad=True)
        else: # ssm compute
            self.inter_proj = nn.Sequential(
                nn.Linear(embed_dim, ssm_dim, bias=bias),
                nn.Linear(ssm_dim, embed_dim, bias=bias)
            )
            
            f = repeat(torch.arange(1, expand_dim + 1), 'k -> k d', d=embed_dim)
            self.f = nn.Parameter(torch.log(f))
            
            self.e_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
            self.s_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
            
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    @staticmethod
    def _build_slope_tensor(d: int):

        def get_slopes(n):

            def get_slopes_power_of_2(n):
                start = 2**(-(2**-(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(
                    n
                )  # In the paper, we only train models that have 2^a heads for some a. This function has
            else:  # some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2**math.floor(
                    math.log2(n)
                )  # when the number of heads is not a power of 2, we use this workaround.
                return (get_slopes_power_of_2(closest_power_of_2) + get_slopes(
                    2 * closest_power_of_2)[0::2][:n - closest_power_of_2])

        # h
        slopes = torch.tensor(get_slopes(d))

        return slopes
    
    def prepare(self, x):
        k = self.expand_dim
        d = self.embed_dim
        b, n = x.shape[:2]
        
        i = self.i_proj(x)
        
        if self.c_type == 1:
            if self.e_type == 1:
                e = self.e_proj(x)
            else:
                e = self.e
                
            if self.s_type == 1:
                s = self.s_proj(x)
            else:
                s = self.s
            
            if self.f_type == 1:
                if self.f_type == 1:
                    # d, k -> k d
                    f1 = F.logsigmoid(self.f1_proj(x)) / self.tau
                    f2 = F.logsigmoid(self.f2_proj(x)) / self.tau
                    f = torch.exp(torch.einsum("... d, ... k -> ... k d", f1, f2))
                elif self.f_type == 2:
                    # 1 d
                    f = torch.exp(F.logsigmoid(self.self.f_proj(x)) / self.tau)
                    f = repeat(f, "... d -> ... k d", k=k)
                elif self.f_type == 3:
                    # k 1
                    f = torch.exp(F.logsigmoid(self.self.f_proj(x)) / self.tau)
                    f = repeat(f, "... k -> ... k d", d=d)
                elif self.f_type == 4: # data independent
                    # k d
                    if self.f_learned:
                        f = F.logsigmoid(self.f) / self.tau
                    else:
                        f = self.f
                    f = torch.exp(f)
                    # f = repeat(f, "... -> b n ...", b=b, n=n)
                elif self.f_type == 5:
                    # k
                    if self.f_learned:
                        f = F.logsigmoid(self.f) / self.tau
                    else:
                        f = self.f
                    f = torch.exp(f)
                    # f = repeat(f, "k -> b n k d", b=b, n=n, d=d)
                    f = repeat(f, "k -> k d", d=d)
                elif self.f_type == 6:
                    # d
                    if self.f_learned:
                        f = F.logsigmoid(self.f) / self.tau
                    else:
                        f = self.f
                    f = torch.exp(f)
                    # f = repeat(f, "d -> b n k d", b=b, n=n, k=k)
                    f = repeat(f, "d -> k d", k=k)
                elif self.f_type == 7:
                    # k, k d -> k d
                    f = torch.einsum("... k, ... k d -> ... k d", self.f_proj(x), self.f)
                    f = torch.exp(F.logsigmoid(f) / self.tau)
                elif self.f_type == 8:
                    # d, k d -> k d
                    f = torch.einsum("... d, ... k d -> ... k d", self.f_proj(x), self.f)
                    f = torch.exp(F.logsigmoid(f) / self.tau)
                elif self.f_type == 9:
                    # d, k -> k d
                    f = torch.einsum("... d, ... k -> ... k d", self.f_proj(x), self.f)
                    f = torch.exp(F.logsigmoid(f) / self.tau)
                elif self.f_type == 10:
                    # k, d -> k d
                    f = torch.einsum("... k, ... d -> ... k d", self.f_proj(x), self.f)
                    f = torch.exp(F.logsigmoid(f) / self.tau)
        else:
            # b n d
            inter_state = F.softplus(self.inter_proj(x))
            i = inter_state * i
            # k d
            f_di = -torch.exp(self.f.float())
            f = torch.exp(torch.einsum("... d, ... k d -> ... k d", inter_state, f_di))
            # b n k
            e = self.e_proj(x)
            # b n k
            s = self.s_proj(x)
        
        return i, e, f, s

    def forward(self, x):
        i, e, f, s = self.prepare(x)
        
        o = pscan_block(i, e, f, s)
        
        o = self.o_proj(o)

        return o