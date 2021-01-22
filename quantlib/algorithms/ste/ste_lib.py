import torch


class STEActKernel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x_in, eps, q_min, q_max):

        x_out = x_in / eps
        x_out = torch.round(x_out.clamp(q_min, q_max))
        x_out = x_out * eps

        return x_out

    @staticmethod
    def backward(ctx, grad_in):
        return grad_in, None, None, None
