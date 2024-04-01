import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_gpu_arch_flags():
    try:
        major = torch.cuda.get_device_capability()[0]
        return [f"-gencode=arch=compute_{major}0,code=sm_{major}0"]
    except Exception as e:
        print(f"Error while detecting GPU architecture: {e}")
        return []


arch_flags = get_gpu_arch_flags()

setup(
    name="lcsm_pytorch",
    packages=find_packages(
        exclude=[
            "tests",
            "benchmarks",
        ]
    ),
    ext_modules=[
        CUDAExtension(
            "pscan_cuda",
            sources=[
                "lcsm_pytorch/ops/cuda/pscan_cuda_kernel.cu",
                "lcsm_pytorch/ops/cuda/pscan_cuda.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O2", "-std=c++14", "-D_GLIBCXX_USE_CXX11_ABI=0"],
                "nvcc": ["-O2", "-std=c++14", "-D_GLIBCXX_USE_CXX11_ABI=0"]
                + arch_flags,
            },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=False),
    },
    install_requires=["torch", "einops", "triton"],
    version="0.0.0",
    author="Doraemonzzz",
    include_package_data=True,
)
