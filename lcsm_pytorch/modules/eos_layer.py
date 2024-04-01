import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from lcsm_pytorch.ops import pscan_cuda_fn
from lcsm_pytorch.utils import print_module


class EosLayer(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        expand_dim=8,
        bias=False,
        c_type=0,  # compute type, 0: ssm, 1: linear layer
        e_type=0,
        o_type=0,
        o_learned=True,
        s_type=0,
        t_type=0,  # transform(act function) type
        ssm_dim=16,
        tau=16,
        use_tau=True,
        **kwargs,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.expand_dim = expand_dim

        self.c_type = c_type
        self.e_type = e_type
        self.o_type = o_type
        self.s_type = s_type
        self.o_learned = o_learned
        self.t_type = t_type

        self.i_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.tau = tau
        self.use_tau = use_tau

        if self.c_type == 1:  # linear
            ##### expand
            if self.e_type == 1:
                self.e_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
            else:
                self.e = nn.Parameter(
                    torch.randn(expand_dim, embed_dim) * 0.1, requires_grad=True
                )

            ##### forget
            if self.o_type == 0:
                if self.o_learned:
                    o = torch.randn(expand_dim, embed_dim) * 0.1
                else:
                    o = self._build_slope_tensor(expand_dim * embed_dim).reshape(
                        expand_dim, -1
                    )
                # k d
                self.o = nn.Parameter(o, requires_grad=o_learned)
            elif self.o_type == 1:  # data dependent
                # d, k -> k d
                self.o1_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
                self.o2_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
                if not self.use_tau:
                    self.gamma_o1 = nn.Parameter(
                        self._build_slope_tensor(embed_dim), requires_grad=False
                    )
                    self.gamma_o2 = nn.Parameter(
                        self._build_slope_tensor(expand_dim), requires_grad=False
                    )
            elif self.o_type == 2:
                # 1 d
                self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
                if not self.use_tau:
                    self.gamma_o = nn.Parameter(
                        self._build_slope_tensor(embed_dim), requires_grad=False
                    )
            elif self.o_type == 3:
                # k 1
                self.o_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
                if not self.use_tau:
                    self.gamma_o = nn.Parameter(
                        self._build_slope_tensor(expand_dim), requires_grad=False
                    )
            elif self.o_type == 4:  # data independent
                if self.o_learned:
                    o = torch.randn(expand_dim) * 0.1
                else:
                    o = self._build_slope_tensor(expand_dim)
                # k 1
                self.o = nn.Parameter(o, requires_grad=self.o_learned)
            elif self.o_type == 5:
                if self.o_learned:
                    o = torch.randn(embed_dim) * 0.1
                else:
                    o = self._build_slope_tensor(embed_dim)
                # 1 d
                self.o = nn.Parameter(o, requires_grad=True)
            elif self.o_type == 6:  # data independent + data dependent
                # k, k d -> k d
                self.o_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
                self.o = nn.Parameter(
                    torch.randn(expand_dim, embed_dim) * 0.1, requires_grad=True
                )
            elif self.o_type == 7:
                # d, k d -> k d
                self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
                self.o = nn.Parameter(
                    torch.randn(expand_dim, embed_dim) * 0.1, requires_grad=True
                )
            elif self.o_type == 8:
                # d, k -> k d
                self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
                self.o = nn.Parameter(torch.randn(expand_dim) * 0.1, requires_grad=True)
            elif self.o_type == 9:
                # k, d -> k d
                self.o_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
                self.o = nn.Parameter(torch.randn(embed_dim) * 0.1, requires_grad=True)
            elif self.o_type == 10:
                # no learn
                self.o = nn.Parameter(
                    torch.ones(expand_dim, embed_dim), requires_grad=False
                )
            elif self.o_type == 11:
                assert self.e_type == 1 and self.s_type == 1
                # complex
                self.o = nn.Parameter(
                    torch.ones(expand_dim * 2, embed_dim), requires_grad=False
                )
                theta = 10000 ** (-2 / embed_dim * torch.arange(expand_dim)).reshape(
                    1, -1
                )
                self.theta = nn.Parameter(theta, requires_grad=False)
                self.theta_cache = torch.empty(0)
            elif self.o_type == 12:
                # cum softmax
                assert self.e_type == 1 and self.s_type == 1
                self.o_proj = nn.Linear(embed_dim, expand_dim, bias=bias)

            ##### shrink
            if self.s_type == 1:
                self.s_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
            else:
                self.s = nn.Parameter(
                    torch.randn(expand_dim, embed_dim) * 0.1, requires_grad=True
                )
        else:  # ssm compute
            self.inter_proj = nn.Sequential(
                nn.Linear(embed_dim, ssm_dim, bias=bias),
                nn.Linear(ssm_dim, embed_dim, bias=bias),
            )

            o = repeat(torch.arange(1, expand_dim + 1), "k -> k d", d=embed_dim)
            self.o = nn.Parameter(torch.log(o))

            self.e_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
            self.s_proj = nn.Linear(embed_dim, expand_dim, bias=bias)

        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    @staticmethod
    def _build_slope_tensor(d: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(
                    n
                )  # In the paper, we only train models that have 2^a heads for some a. This function has
            else:  # some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2 ** math.floor(
                    math.log2(n)
                )  # when the number of heads is not a power of 2, we use this workaround.
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        # h
        slopes = -torch.tensor(get_slopes(d))

        return slopes

    def transform(self, x, eps=10):
        if self.t_type == 0:
            return x
        elif self.t_type == 1:
            return F.relu(x)
        elif self.t_type == 2:
            return F.sigmoid(x)
        elif self.t_type == 3:
            return 1 + F.elu(x)
        elif self.t_type == 4:
            return F.silu(x)
        elif self.t_type == 5:
            return F.elu(x)
        elif self.t_type == 6:
            return F.relu(x) ** 2
        elif self.t_type == 7:
            return x**2

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

            e = self.transform(e)
            s = self.transform(s)

            if self.o_type == 0:
                # k d
                if self.o_learned:
                    if self.use_tau:
                        o = F.logsigmoid(self.o) / self.tau
                    else:
                        o = -self.o
                else:
                    o = -self.o
                o = torch.exp(o)

            elif self.o_type == 1:
                # d, k -> k d
                if self.use_tau:
                    o1 = F.logsigmoid(self.o1_proj(x)) / self.tau
                    o2 = F.logsigmoid(self.o2_proj(x)) / self.tau
                else:
                    o1 = F.sigmoid(self.o1_proj(x)) * self.gamma_o1
                    o2 = F.sigmoid(self.o2_proj(x)) * self.gamma_o2
                # ... d, ... k -> ... k d
                o = torch.exp(o1.unsqueeze(-2) + o2.unsqueeze(-1))
            elif self.o_type == 2:
                # 1 d
                if self.use_tau:
                    o = torch.exp(o.logsigmoid(self.o_proj(x)) / self.tau)
                else:
                    o = torch.exp(o.sigmoid(self.o_proj(x)) * self.gamma_f)
                o = repeat(o, "... d -> ... k d", k=k)
            elif self.o_type == 3:
                # k 1
                if self.use_tau:
                    o = torch.exp(o.logsigmoid(self.o_proj(x)) / self.tau)
                else:
                    o = torch.exp(o.sigmoid(self.o_proj(x)) * self.gamma_f)
                o = repeat(o, "... k -> ... k d", d=d)
                # o = repeat(o, "... -> b n ...", b=b, n=n)
            elif self.o_type == 4:  # data independent
                # k
                if self.o_learned:
                    o = F.logsigmoid(self.o) / self.tau
                else:
                    o = self.o
                o = torch.exp(o)
                # o = repeat(o, "k -> b n k d", b=b, n=n, d=d)
                o = repeat(o, "k -> k d", d=d)
            elif self.o_type == 5:
                # d
                if self.o_learned:
                    o = F.logsigmoid(self.o) / self.tau
                else:
                    o = self.o
                o = torch.exp(o)
                # o = repeat(o, "d -> b n k d", b=b, n=n, k=k)
                o = repeat(o, "d -> k d", k=k)
            elif self.o_type == 6:
                # k, k d -> k d
                o = torch.einsum("... k, ... k d -> ... k d", self.o_proj(x), self.o)
                o = torch.exp(o.logsigmoid(o) / self.tau)
            elif self.o_type == 7:
                # d, k d -> k d
                o = torch.einsum("... d, ... k d -> ... k d", self.o_proj(x), self.o)
                o = torch.exp(o.logsigmoid(o) / self.tau)
            elif self.o_type == 8:
                # d, k -> k d
                o = torch.einsum("... d, ... k -> ... k d", self.o_proj(x), self.o)
                o = torch.exp(o.logsigmoid(o) / self.tau)
            elif self.o_type == 9:
                # k, d -> k d
                o = torch.einsum("... k, ... d -> ... k d", self.o_proj(x), self.o)
                o = torch.exp(o.logsigmoid(o) / self.tau)
            elif self.o_type == 10:
                # no f
                o = self.o
            elif self.o_type == 11:
                # complex
                if self.theta_cache.shape[0] == 0:
                    b, n, d = x.shape
                    index = (
                        torch.arange(n).reshape(-1, 1).to(torch.float32).to(x.device)
                    )
                    self.theta_cache = index * self.theta
                theta = self.theta_cache
                e = torch.cat([e * torch.cos(theta), e * torch.sin(theta)], dim=-1).to(
                    x.dtype
                )
                s = torch.cat([s * torch.cos(theta), s * torch.sin(theta)], dim=-1).to(
                    x.dtype
                )
                o = self.o
            elif self.o_type == 12:
                # cum softmax
                f_ = torch.clamp(self.o_proj(x), min=-10, max=10).to(torch.float32)
                f_logexpcumsum = torch.cat(
                    [-1e5 * torch.ones(b, 1, k).to(x), torch.logcumsumexp(f_, dim=1)],
                    dim=1,
                )
                o = torch.exp(f_logexpcumsum[:, :n] - f_logexpcumsum[:, 1:])
                e = (1 - f) * e
                o = repeat(o, "... k -> ... k d", d=d)
                i = i.to(torch.float32)
                s = s.to(torch.float32)
        else:
            # b n d
            inter_state = F.softplus(self.inter_proj(x))
            i = inter_state * i
            # k d
            f_di = -torch.exp(self.o.float())
            o = torch.exp(torch.einsum("... d, ... k d -> ... k d", inter_state, f_di))
            # b n k
            e = self.e_proj(x)
            # b n k
            s = self.s_proj(x)

        return i, e, f, s

    def extra_repr(self):
        return print_module(self)

    def forward(self, x):
        i, e, f, s = self.prepare(x)
        y = pscan_cuda_fn(i, e, f, s).to(x.dtype)

        y = self.norm(y)

        y = self.output_proj(y)

        return y
