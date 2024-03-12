import torch


def is_dependent(x):
    return len(x.shape) >= 3


def process(x, i):
    if is_dependent(x):
        return x[:, i]
    else:
        return x


def complex_log(x, eps=1e-6):
    eps = x.new_tensor(eps)
    real = x.abs().maximum(eps).log()
    imag = (x < 0).to(x.dtype) * torch.pi
    return torch.complex(real.to(torch.float), imag.to(torch.float))
