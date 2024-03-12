import torch
import torch.nn.functional as F


def naively_compute_sequentially(coeffs, values, init):
    x = [init]  # x_0
    for a, b in zip(coeffs, values):
        x.append(a * x[-1] + b)
    return torch.stack(x)


device = "cuda:0"  # change as necessary
seq_len = 10000  # change as you wish

# Generate some random input data:
coeffs = torch.randn(seq_len, device=device)
values = torch.randn(seq_len, device=device)  # includes initial value
init = torch.randn(1, device=device)

# Compute the sequence:
x_naive = naively_compute_sequentially(coeffs, values, init)  # includes initial value


def complex_log(float_input, eps=1e-6):
    eps = float_input.new_tensor(eps)
    real = float_input.abs().maximum(eps).log()
    imag = (float_input < 0).to(float_input.dtype) * torch.pi
    return torch.complex(real, imag)


def compute_in_parallel(coeffs, values, init):
    log_coeffs = complex_log(coeffs)
    log_values = complex_log(values)
    a_star = F.pad(torch.cumsum(log_coeffs, dim=-1), (1, 0))  # eq (2) in paper
    log_x0_plus_b_star = torch.logcumsumexp(
        log_values - a_star, dim=-1
    )  # eq (7) in paper
    log_x = a_star + log_x0_plus_b_star  # eq (1) in paper
    return torch.exp(log_x).real


# Compute the sequence:
x = compute_in_parallel(coeffs, values)  # includes initial value

print(torch.norm(x - x_naive))
