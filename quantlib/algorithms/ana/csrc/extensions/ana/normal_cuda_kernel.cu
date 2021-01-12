#include <torch/extension.h>
#include <vector>

// #include <stdio.h>  // for debug

#include <cuda.h>
#include <cuda_runtime.h>


#define THREADS_PER_BLOCK 1024

#define PI 3.141592653589793


// GPU kernels (vanilla)

template <typename scalar_t>
__global__ void normal_forward_cuda_kernel(
    scalar_t * const __restrict__ x_out,
    const scalar_t * __restrict__ x_in,
    const int64_t len_x,
    const scalar_t * __restrict__ t,
    const int64_t len_t,
    const scalar_t * __restrict__ q,
    const scalar_t * __restrict__ s_for,
    const scalar_t * __restrict__ training
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < len_x)
    {
        float sum = q[0];

        for (int it = 0; it < len_t; ++it)
        {
            // input position relative to the threshold
            float x_minus_t = x_in[ix] - t[it];

            // expected value of the Heaviside function is the CDF of the normal distribution
            float cdf;
            if (training && (*s_for != 0.0f))
            {
                float s_inv = 1.0f / (*s_for);
                float x_minus_t_over_s = x_minus_t * s_inv;
                cdf = (float) normcdf((double) x_minus_t_over_s);
            }
            else
            {
                cdf = (float) (x_minus_t >= 0.0f); // use the Heaviside which maps zero to one
            }

            // dilate and accumulate expected step value
            float dq = q[it + 1] - q[it];
            sum += dq * cdf;
        }

        x_out[ix] = sum;
    }
    else  // I am out of bounds!
    {
        return;
    }
}


template <typename scalar_t>
__global__ void normal_backward_cuda_kernel(
    scalar_t * const __restrict__ grad_out,
    const scalar_t * __restrict__ grad_in,
    const scalar_t * __restrict__ x_in,
    const int64_t len_x,
    const scalar_t * __restrict__ t,
    const int64_t len_t,
    const scalar_t * __restrict__ q,
    const scalar_t * __restrict__ s_back
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < len_x)
    {
        float sum = 0.0f;

        for (int it = 0; it < len_t; ++it)
        {
            // input position relative to the threshold
            float x_minus_t  = x_in[ix] - t[it];

            // the derivative of the expected (i.e., regularised) step function is the PDF of the normal distribution
            float pdf;
            if (*s_back != 0.0f)
            {
                float s_inv = 1.0f / (*s_back);
                float x_minus_t_over_s = x_minus_t * s_inv;
                float exp_x_minus_t_over_s_square = expf(-(x_minus_t_over_s * x_minus_t_over_s) / 2.0f);
                pdf = exp_x_minus_t_over_s_square * s_inv * (1 / sqrt(2 * PI));
            }
            else
            {
                pdf = 0.0f;  // no noise, no gradient!
            }

            // dilate and accumulate expected derivative
            float dq = q[it + 1] - q[it];
            sum += dq * pdf;
        }

        // compose gradients
        grad_out[ix] = sum * grad_in[ix];
    }
    else  // I am out of bounds!
    {
        return;
    }
}


// dispatchers

torch::Tensor normal_forward_cuda_dispatch(
    torch::Tensor x_in,
    torch::Tensor t,
    torch::Tensor q,
    torch::Tensor s_for,
    torch::Tensor training
)
{
    auto x_out = torch::zeros_like(x_in);
    auto len_x = x_in.numel();

    const dim3 blocks((len_x + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES(
        x_in.type(),
        "normal_forward_cuda",
        ([&] {
            normal_forward_cuda_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
                x_out.data_ptr<scalar_t>(),
                x_in.data_ptr<scalar_t>(),
                len_x,
                t.data_ptr<scalar_t>(),
                t.numel(),
                q.data_ptr<scalar_t>(),
                s_for.data_ptr<scalar_t>(),
                training.data_ptr<scalar_t>()
            );
        })
    );

    return x_out;
}


torch::Tensor normal_backward_cuda_dispatch(
    torch::Tensor grad_in,
    torch::Tensor x_in,
    torch::Tensor t,
    torch::Tensor q,
    torch::Tensor s_back
)
{
    auto grad_out = torch::zeros_like(x_in);
    auto len_x = x_in.numel();

    const dim3 blocks((len_x + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES(
        x_in.type(),
        "normal_backward_cuda",
        ([&] {
            normal_backward_cuda_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
                grad_out.data_ptr<scalar_t>(),
                grad_in.data_ptr<scalar_t>(),
                x_in.data_ptr<scalar_t>(),
                len_x,
                t.data_ptr<scalar_t>(),
                t.numel(),
                q.data_ptr<scalar_t>(),
                s_back.data_ptr<scalar_t>()
            );
        })
    );

    return grad_out;
}
