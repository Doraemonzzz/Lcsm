import logging
import os
import sys

import torch
import torch.distributed as dist

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("print_config")


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


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def logging_info(string):
    if is_main_process():
        logger.info(string)


def print_params(**kwargs):
    if is_main_process():
        logger.info(f"start print config of {kwargs['__class__']}")
        for key in kwargs:
            if key in ["__class__", "self"]:
                continue
            logger.info(f"{key}: {kwargs[key]}")
        logger.info(f"end print config of {kwargs['__class__']}")


def print_config(config):
    if is_main_process():
        logger.info(f"start print config of {config['__class__']}")
        for key in config:
            if key in ["__class__", "self"]:
                continue
            logger.info(f"{key}: {config[key]}")
        logger.info(f"end print config of {config['__class__']}")


def print_module(module):
    named_modules = set()
    for p in module.named_modules():
        named_modules.update([p[0]])
    named_modules = list(named_modules)

    string_repr = ""
    for p in module.named_parameters():
        name = p[0].split(".")[0]
        if name not in named_modules:
            string_repr = (
                string_repr
                + "("
                + name
                + "): "
                + "Tensor("
                + str(tuple(p[1].shape))
                + ", requires_grad="
                + str(p[1].requires_grad)
                + ")\n"
            )

    return string_repr.rstrip("\n")
