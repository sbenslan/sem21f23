#include <torch/extension.h>
#include <vector>

// #include <stdio.h>  // for debug

#include <cuda.h>
#include <cuda_runtime.h>


#define THREADS_PER_BLOCK 1024

#define SIGN(x) ((x < 0.0f) ? -1.0f : 1.0f)
#define ABS(x) ((x < 0.0f) ? -x : x)


// GPU kernels (vanilla)

template <typename scalar_t>
__global__ void triangular_forward_cuda_kernel(
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

            // expected value of the Heaviside function is the CDF of the triangular distribution
            float cdf;
            if (training && (*s_for != 0.0f))
            {
                float s_inv = 1.0f / (*s_for);
                float x_minus_t_over_s = x_minus_t * s_inv;
                cdf = ((ABS(x_minus_t_over_s) > 1.0f) ? SIGN(x_minus_t_over_s) : (x_minus_t_over_s * (2 - ABS(x_minus_t_over_s))));
                cdf = (cdf + 1.0f) / 2;
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
__global__ void triangular_backward_cuda_kernel(
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

            // the derivative of the expected (i.e., regularised) step function is the PDF of the triangular distribution
            float pdf;
            if (*s_back != 0.0f)
            {
                float s_inv = 1.0f / (*s_back);
                float abs_x_minus_t_over_s = ABS(x_minus_t * s_inv);
                pdf = ((abs_x_minus_t_over_s > 1.0f) ? 0.0f : (1.0f - abs_x_minus_t_over_s) * s_inv);
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

torch::Tensor triangular_forward_cuda_dispatch(
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
        "triangular_forward_cuda",
        ([&] {
            triangular_forward_cuda_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
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


torch::Tensor triangular_backward_cuda_dispatch(
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
        "triangular_backward_cuda",
        ([&] {
            triangular_backward_cuda_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
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
