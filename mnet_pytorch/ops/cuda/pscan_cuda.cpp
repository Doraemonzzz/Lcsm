#include <torch/extension.h>

torch::Tensor pscan_forward_cuda(
    torch::Tensor &x,
    torch::Tensor &lambda,
    bool is_dd // whether lambda is data dependent
);

std::vector<torch::Tensor> pscan_backward_cuda(
    torch::Tensor &x,
    torch::Tensor &lambda,
    torch::Tensor &hidden_states,
    torch::Tensor &grad_output,
    bool is_dd, // whether lambda is data dependent
    bool learned // whether learned lambda or not
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &pscan_forward_cuda, "Pscan forward (CUDA)");
  m.def("backward", &pscan_backward_cuda, "Pscan backward (CUDA)");
}
