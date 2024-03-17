import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from mnet_pytorch.ops import pscan_cuda_fn
from mnet_pytorch.utils import print_module


class EOS(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        expand_dim=8,
        bias=False,
        c_type=0,  # compute type, 1: linear layer 2: ssm
        e_type=0,
        f_type=0,
        s_type=0,
        f_learned=True,
        ssm_dim=16,
        tau=16,
        t_type=0,  # transform type
        use_tau=True,
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
            if self.f_type == 0:
                if self.f_learned:
                    f = torch.randn(expand_dim, embed_dim) * 0.1
                else:
                    f = self._build_slope_tensor(expand_dim * embed_dim).reshape(
                        expand_dim, -1
                    )
                # k d
                self.f = nn.Parameter(f, requires_grad=f_learned)
            elif self.f_type == 1:  # data dependent
                # d, k -> k d
                self.f1_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
                self.f2_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
                if not self.use_tau:
                    self.gamma_f1 = nn.Parameter(
                        self._build_slope_tensor(embed_dim), requires_grad=False
                    )
                    self.gamma_f2 = nn.Parameter(
                        self._build_slope_tensor(expand_dim), requires_grad=False
                    )
            elif self.f_type == 2:
                # 1 d
                self.f_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
                if not self.use_tau:
                    self.gamma_f = nn.Parameter(
                        self._build_slope_tensor(embed_dim), requires_grad=False
                    )
            elif self.f_type == 3:
                # k 1
                self.f_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
                if not self.use_tau:
                    self.gamma_f = nn.Parameter(
                        self._build_slope_tensor(expand_dim), requires_grad=False
                    )
            elif self.f_type == 4:  # data independent
                if self.f_learned:
                    f = torch.randn(expand_dim) * 0.1
                else:
                    f = self._build_slope_tensor(expand_dim)
                # k 1
                self.f = nn.Parameter(f, requires_grad=self.f_learned)
            elif self.f_type == 5:
                if self.f_learned:
                    f = torch.randn(embed_dim) * 0.1
                else:
                    f = self._build_slope_tensor(embed_dim)
                # 1 d
                self.f = nn.Parameter(torch.randn(embed_dim) * 0.1, requires_grad=True)
            elif self.f_type == 6:  # data independent + data dependent
                # k, k d -> k d
                self.f_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
                self.f = nn.Parameter(
                    torch.randn(expand_dim, embed_dim) * 0.1, requires_grad=True
                )
            elif self.f_type == 7:
                # d, k d -> k d
                self.f_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
                self.f = nn.Parameter(
                    torch.randn(expand_dim, embed_dim) * 0.1, requires_grad=True
                )
            elif self.f_type == 8:
                # d, k -> k d
                self.f_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
                self.f = nn.Parameter(torch.randn(expand_dim) * 0.1, requires_grad=True)
            elif self.f_type == 9:
                # k, d -> k d
                self.f_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
                self.f = nn.Parameter(torch.randn(embed_dim) * 0.1, requires_grad=True)
            elif self.f_type == 10:
                # no learn
                self.f = nn.Parameter(
                    torch.ones(expand_dim, embed_dim), requires_grad=False
                )
            elif self.f_type == 11:
                assert self.e_type == 1 and self.s_type == 1
                # complex
                self.f = nn.Parameter(
                    torch.ones(expand_dim * 2, embed_dim), requires_grad=False
                )
                theta = 10000 ** (-2 / embed_dim * torch.arange(expand_dim)).reshape(
                    1, -1
                )
                self.theta = nn.Parameter(theta, requires_grad=False)
                self.theta_cache = torch.empty(0)

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

            f = repeat(torch.arange(1, expand_dim + 1), "k -> k d", d=embed_dim)
            self.f = nn.Parameter(torch.log(f))

            self.e_proj = nn.Linear(embed_dim, expand_dim, bias=bias)
            self.s_proj = nn.Linear(embed_dim, expand_dim, bias=bias)

        self.norm = nn.LayerNorm(embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

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

            if self.f_type == 0:
                # k d
                if self.f_learned:
                    if self.use_tau:
                        f = F.logsigmoid(self.f) / self.tau
                    else:
                        f = -self.f
                else:
                    f = -self.f
                f = torch.exp(f)

            elif self.f_type == 1:
                # d, k -> k d
                if self.use_tau:
                    f1 = F.logsigmoid(self.f1_proj(x)) / self.tau
                    f2 = F.logsigmoid(self.f2_proj(x)) / self.tau
                else:
                    f1 = F.sigmoid(self.f1_proj(x)) * self.gamma_f1
                    f2 = F.sigmoid(self.f2_proj(x)) * self.gamma_f2
                # ... d, ... k -> ... k d
                f = torch.exp(f1.unsqueeze(-2) + f2.unsqueeze(-1))
            elif self.f_type == 2:
                # 1 d
                if self.use_tau:
                    f = torch.exp(F.logsigmoid(self.f_proj(x)) / self.tau)
                else:
                    f = torch.exp(F.sigmoid(self.f_proj(x)) * self.gamma_f)
                f = repeat(f, "... d -> ... k d", k=k)
            elif self.f_type == 3:
                # k 1
                if self.use_tau:
                    f = torch.exp(F.logsigmoid(self.f_proj(x)) / self.tau)
                else:
                    f = torch.exp(F.sigmoid(self.f_proj(x)) * self.gamma_f)
                f = repeat(f, "... k -> ... k d", d=d)
                # f = repeat(f, "... -> b n ...", b=b, n=n)
            elif self.f_type == 4:  # data independent
                # k
                if self.f_learned:
                    f = F.logsigmoid(self.f) / self.tau
                else:
                    f = self.f
                f = torch.exp(f)
                # f = repeat(f, "k -> b n k d", b=b, n=n, d=d)
                f = repeat(f, "k -> k d", d=d)
            elif self.f_type == 5:
                # d
                if self.f_learned:
                    f = F.logsigmoid(self.f) / self.tau
                else:
                    f = self.f
                f = torch.exp(f)
                # f = repeat(f, "d -> b n k d", b=b, n=n, k=k)
                f = repeat(f, "d -> k d", k=k)
            elif self.f_type == 6:
                # k, k d -> k d
                f = torch.einsum("... k, ... k d -> ... k d", self.f_proj(x), self.f)
                f = torch.exp(F.logsigmoid(f) / self.tau)
            elif self.f_type == 7:
                # d, k d -> k d
                f = torch.einsum("... d, ... k d -> ... k d", self.f_proj(x), self.f)
                f = torch.exp(F.logsigmoid(f) / self.tau)
            elif self.f_type == 8:
                # d, k -> k d
                f = torch.einsum("... d, ... k -> ... k d", self.f_proj(x), self.f)
                f = torch.exp(F.logsigmoid(f) / self.tau)
            elif self.f_type == 9:
                # k, d -> k d
                f = torch.einsum("... k, ... d -> ... k d", self.f_proj(x), self.f)
                f = torch.exp(F.logsigmoid(f) / self.tau)
            elif self.f_type == 10:
                # no f
                f = self.f
            elif self.f_type == 11:
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
                f = self.f
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

    def extra_repr(self):
        return print_module(self)

    def forward(self, x):
        i, e, f, s = self.prepare(x)
        o = pscan_cuda_fn(i, e, f, s)

        o = self.norm(o)

        o = self.o_proj(o)

        return o
