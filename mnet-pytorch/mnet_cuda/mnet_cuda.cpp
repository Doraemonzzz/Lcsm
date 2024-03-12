#include <torch/extension.h>

torch::Tensor mnet_forward_cuda(torch::Tensor &i, torch::Tensor &e, torch::Tensor &f, torch::Tensor &s);

torch::Tensor mnet_backward_cuda(torch::Tensor &grad_output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mnet_forward_cuda, "Mnet forward (CUDA)");
  m.def("backward", &mnet_backward_cuda, "Mnet backward (CUDA)");
}