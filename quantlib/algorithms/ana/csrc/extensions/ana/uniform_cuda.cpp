#include <torch/extension.h>
#include <vector>

// #include <stdio.h>  // for debug


// declarations of CUDA interface ("dispatchers")

torch::Tensor uniform_forward_cuda_dispatch(
    torch::Tensor x_in,
    torch::Tensor t,
    torch::Tensor q,
    torch::Tensor s_for,
    torch::Tensor training
);

torch::Tensor uniform_backward_cuda_dispatch(
    torch::Tensor grad_in,
    torch::Tensor x_in,
    torch::Tensor t,
    torch::Tensor q,
    torch::Tensor s_back
);


// definitions of C++ interface

// consistency checks
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor uniform_forward_cuda(
    torch::Tensor x_in,
    torch::Tensor t,
    torch::Tensor q,
    torch::Tensor s_for,
    torch::Tensor training
)
{
    CHECK_INPUT(x_in);
    CHECK_INPUT(t);
    CHECK_INPUT(q);
    CHECK_INPUT(s_for);
    CHECK_INPUT(training);

    return uniform_forward_cuda_dispatch(x_in, t, q, s_for, training);
}


torch::Tensor uniform_backward_cuda(
    torch::Tensor grad_in,
    torch::Tensor x_in,
    torch::Tensor t,
    torch::Tensor q,
    torch::Tensor s_back
)
{
    CHECK_INPUT(grad_in);
    CHECK_INPUT(x_in);
    CHECK_INPUT(t);
    CHECK_INPUT(q);
    CHECK_INPUT(s_back);

    return uniform_backward_cuda_dispatch(grad_in, x_in, t, q, s_back);
}


// compile into a Python module

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &uniform_forward_cuda, "ANA uniform noise forward (CUDA)");
    m.def("backward", &uniform_backward_cuda, "ANA uniform noise backward (CUDA)");
}
