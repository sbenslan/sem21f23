import torch


def forward(x_in, t, q, s_for, training):

    t_shape = [t.numel()] + [1 for _ in range(x_in.dim())]
    x_minus_t = x_in - t.reshape(t_shape)

    if training and s_for != 0.:
        x_minus_t_over_s = torch.abs(x_minus_t / s_for)
        cdf_temp = (torch.sign(x_minus_t) * x_minus_t_over_s * (2 - x_minus_t_over_s) + 1) / 2
        cond = (x_minus_t_over_s <= 1.)
        cdf = torch.zeros_like(x_minus_t)
        cdf[cond] = cdf_temp[cond]
        cdf[~cond] = (x_minus_t >= 0.).float()[~cond]
    else:
        cdf = (x_minus_t >= 0.).float()

    d = q[1:] - q[:-1]
    x_out = q[0] + torch.sum(d.reshape(t_shape) * cdf, 0)

    return x_out


def backward(grad_in, x_in, t, q, s_back):

    t_shape = [t.numel()] + [1 for _ in range(x_in.dim())]
    x_minus_t = x_in - t.reshape(t_shape)

    if s_back != 0.:
        x_minus_t_over_s = torch.abs(x_minus_t / s_back)
        pdf = torch.max(torch.tensor([0.]), 1 - x_minus_t_over_s) / s_back
    else:
        pdf = torch.zeros_like(x_minus_t)

    d = q[1:] - q[:-1]
    local_jacobian = torch.sum(d.reshape(t_shape) * pdf, 0)
    grad_out = grad_in * local_jacobian

    return grad_out
