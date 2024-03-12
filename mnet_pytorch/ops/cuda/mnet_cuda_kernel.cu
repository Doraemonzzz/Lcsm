
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF_AND_BF16(...)   \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \

#define AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BF16(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                        \
      TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF_AND_BF16(__VA_ARGS__))



// CUDA kernel for forward pass
template <typename scalar_t>
__global__ void mnet_forward_kernel(
    const scalar_t* i, const scalar_t* e, const scalar_t* f, const scalar_t* s,
    const scalar_t* m, const scalar_t* o,
    int64_t b, int64_t n, int64_t k, int64_t d
) {
    // Compute the global indices of the current thread
    // batch
    int64_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    // feature
    int64_t idz = blockIdx.x * blockDim.x + threadIdx.x;


    if (idy < b && idz < d) {
        scalar_t hidden_state = 0;
        #pragma unroll
        for (int64_t idx = 0; idx < n; ++idx) {
            int64_t index = idx * b * d + idy * d + idz;
            for (int64_t idu = 0; idu < k; ++idu) {
                m =
            }
            hidden_state = lambda[index] * hidden_state + x[index];
            output[index] = hidden_state;
        }
    }
}

// CUDA kernel for backward pass
template <typename scalar_t>
__global__ void mnet_backward_kernel(
    const scalar_t* x, const scalar_t* lambda, const scalar_t* hidden_states, const scalar_t* grad_output,
    scalar_t* grad_x, scalar_t* grad_lambda,
    int64_t n, int64_t b, int64_t d) {
    // batch
    int64_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    // feature
    int64_t idz = blockIdx.x * blockDim.x + threadIdx.x;

    if (idy < b && idz < d) {
        scalar_t grad_hidden_state = 0;
        #pragma unroll
        for (int64_t idx = n - 1; idx >= 0; --idx) {
            int64_t index = idx * b * d + idy * d + idz;
            int64_t j = ((idx == n - 1) ? 0 : index + b * d);
            grad_hidden_state = grad_output[index] + lambda[j] * grad_hidden_state;

            grad_lambda[index] = grad_hidden_state * ((idx == 0) ? scalar_t(0) : hidden_states[index - b * d]);
            grad_x[index] = grad_hidden_state;
        }
    }
}

torch::Tensor mnet_forward_cuda(torch::Tensor &i, torch::Tensor &e, torch::Tensor &f, torch::Tensor &s) {
    auto o = torch::zeros_like(i);
    auto m = torch::zeros({b, 1, k, d});
    const int64_t b = i.size(0);
    const int64_t n = i.size(1);
    const int64_t d = i.size(2);
    const int64_t k = e.size(-1);

    dim3 threads(128, 8);
    dim3 blocks((d + threads.x - 1) / threads.x, (b + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BF16(x.scalar_type(), "mnet_forward_cuda", ([&] {
        mnet_forward_kernel<scalar_t><<<blocks, threads>>>(
            i.data_ptr<scalar_t>(), e.data_ptr<scalar_t>(), f.data_ptr<scalar_t>(), s.data_ptr<scalar_t>(),
            m.data_ptr<scalar_t>(), o.data_ptr<scalar_t>(), b, n, k, d);
    }));

    return o;
}

std::vector<torch::Tensor> mnet_backward_cuda(
    torch::Tensor &x, torch::Tensor &lambda, torch::Tensor &hidden_states, torch::Tensor &grad_output) {
    auto grad_x = torch::zeros_like(x);
    auto grad_lambda = torch::zeros_like(lambda);

    const int64_t n = x.size(0);
    const int64_t b = x.size(1);
    const int64_t d = x.size(2);

    dim3 threads(128, 8);
    dim3 blocks((d + threads.x - 1) / threads.x, (b + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BF16(grad_output.scalar_type(), "mnet_backward_cuda", ([&] {
        mnet_backward_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(), lambda.data_ptr<scalar_t>(), hidden_states.data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(), grad_lambda.data_ptr<scalar_t>(), n, b, d);
    }));

    return {grad_x, grad_lambda};
}
