import torch


def forward(x_in, t, q, s_for, training):

    t_shape = [t.numel()] + [1 for _ in range(x_in.dim())]  # dimensions with size 1 enable broadcasting
    x_minus_t = x_in - t.reshape(t_shape)

    if training and s_for != 0.:
        s_inv = 1 / s_for
        cdf = torch.clamp(0.5 * (x_minus_t * s_inv + 1), 0., 1.)
    else:
        cdf = (x_minus_t >= 0.).float()

    d = q[1:] - q[:-1]
    x_out = q[0] + torch.sum(d.reshape(t_shape) * cdf, 0)

    return x_out


def backward(grad_in, x_in, t, q, s_back):

    t_shape = [t.numel()] + [1 for _ in range(x_in.dim())]  # dimensions with size 1 enable broadcasting
    x_minus_t = x_in - t.reshape(t_shape)

    if s_back != 0.:
        s_inv = 1 / s_back
        pdf = (torch.abs_(x_minus_t) <= s_back).float() * (0.5 * s_inv)
    else:
        pdf = torch.zeros_like(grad_in)

    d = q[1:] - q[:-1]
    local_jacobian = torch.sum(d.reshape(t_shape) * pdf, 0)
    grad_out = grad_in * local_jacobian

    return grad_out
