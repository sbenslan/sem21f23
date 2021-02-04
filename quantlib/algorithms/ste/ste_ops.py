import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from ..controller import Controller
from . import ste_lib


__all__ = [
    'STEController',
    'STEActivationNew',
    'STEActivation',
]


class STEController(Controller):
    def __init__(self, modules, clear_optim_state_on_step=False):
        super(STEController).__init__()
        self.modules = modules
        self.clear_optim_state_on_step = clear_optim_state_on_step

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k in ()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step_pre_training(self, epoch, optimizer=None, tb_writer=None):
        # step each STE module
        for m in self.modules:
            m.step(epoch)
        if (optimizer is not None) and self.clear_optim_state_on_step:
            for m in self.modules:
                if m.quant_start_epoch == epoch:
                    optimizer.state.clear()  # weight decay?

    @staticmethod
    def get_ste_modules(nodes_set):
        return [n[1] for n in nodes_set if isinstance(n[1], STEActivation)]


class STEModule(torch.nn.Module):

    def __init__(self, quantizer_spec):
        super(STEModule, self).__init__()
        STEModule.setup_quantizer(self, quantizer_spec)

    @staticmethod
    def setup_quantizer(stemod, quantizer_spec):

        eps = torch.Tensor([quantizer_spec['eps']])
        stemod.register_parameter('eps', nn.Parameter(eps, requires_grad=False))

        quant_levels = torch.arange(0, 2 ** quantizer_spec['n_bits']).to(dtype=torch.float32)
        if quantizer_spec['signed']:
            quant_levels = quant_levels - 2 ** (quantizer_spec['n_bits'] - 1)
            if quantizer_spec['balanced']:
                quant_levels = quant_levels[1:]
        q_min = quant_levels[0]
        q_max = quant_levels[-1]
        stemod.signed = quantizer_spec['signed']
        stemod.register_parameter('q_min', nn.Parameter(q_min, requires_grad=False))
        stemod.register_parameter('q_max', nn.Parameter(q_max, requires_grad=False))


class STEActivationNew(STEModule):

    def __init__(self, quantizer_spec, quant_start_epoch=0):
        super(STEActivationNew, self).__init__(quantizer_spec)
        self.ste_op = getattr(ste_lib, 'STEActKernel').apply
        self.register_parameter('m', nn.Parameter(torch.ones(1), requires_grad=False))  # maximum absolute input value (i.e., range)

        assert 0 <= quant_start_epoch
        self.quant_start_epoch = quant_start_epoch
        self.is_quantized = (self.quant_start_epoch == 0)
        self.monitor_epoch = self.quant_start_epoch - 1  # during this epoch, statistics about input tensors range will be collected
        self.is_monitoring = (self.quant_start_epoch == 1)

    def step(self, epoch):

        if epoch == self.monitor_epoch:

            self.is_monitoring = True
            self.m.data[0] = 0.0  # initialise range statistic

        elif epoch == self.quant_start_epoch:

            self.is_monitoring = False  # the next epoch, deactivate "monitoring" mode
            self.eps.data = self.m.data / (self.q_max - self.q_min)  # set quantum
            if self.signed:
                self.eps *= 2

            self.is_quantized = True

        elif self.quant_start_epoch < epoch:  # in case of restart; we do not store these variables in the node's state
            self.is_quantized = True

    def forward(self, x):

        if self.is_monitoring:
            self.m.data = torch.max(self.m.data[0], torch.max(torch.abs(x)))

        if self.is_quantized:
            return self.ste_op(x, self.eps, self.q_min, self.q_max)
        else:
            return x


class STEActivation(torch.nn.Module):
    """Quantizes activations according to the straight-through estiamtor (STE).
    Needs a STEController, if `quant_start_epoch` > 0.

    monitor_epoch: In this epoch, keep track of the maximal activation value (absolute value).
        Then (at epoch >= quant_start_epoch), clamp the values to [-max, max], and then do quantization.
        If monitor_epoch is None, max=1 is used."""
    def __init__(self, num_levels=2**8-1, quant_start_epoch=0):
        super(STEActivation, self).__init__()
        assert(num_levels >= 2)
        self.num_levels = num_levels
        self.abs_max_value = torch.nn.Parameter(torch.ones(1), requires_grad=False)

        assert(quant_start_epoch >= 0)
        self.quant_start_epoch = quant_start_epoch
        self.started = self.quant_start_epoch == 0

        self.monitor_epoch = self.quant_start_epoch - 1
        self.monitoring = False
        if 0 <= self.monitor_epoch:
            self.monitoring = self.monitor_epoch == 0

    def step(self, epoch):
        if self.monitor_epoch == epoch:
            self.monitoring = True
            self.abs_max_value.data[0] = 0.0  # prepare to log maximum activation value
        else:
            self.monitoring = False

        if self.quant_start_epoch <= epoch:
            self.started = True

    @staticmethod
    def ste_round_functional(x):
        return x - (x - x.round()).detach()

    def forward(self, x):
        if self.monitoring:
            self.abs_max_value.data[0] = max(self.abs_max_value.item(), x.abs().max())

        if self.started:
            x = x / self.abs_max_value.item()  # map from [-max, max] to [-1, 1]
            xclamp = x.clamp(-1, 1)
            y = xclamp
            y = (y + 1) / 2  # map from [-1,1] to [0,1]
            y = STEActivation.ste_round_functional(y * (self.num_levels - 1)) / (self.num_levels - 1)
            y = 2 * y - 1  # map from [0, 1] to [-1, 1]
            y = y * self.abs_max_value.item()  # map from [-1, 1] to [-max, max]
        else:
            y = x

        return y



class _STEBatchNorm(_BatchNorm):
    def __init__(self, ste_modules, start_epoch, num_levels_mult, num_levels_add, step_mult, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(_STEBatchNorm, self).__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        # ste_modules[0]: the ste module before the activation-quantized layer
        # preceding the batch norm (defining the quantum of input activations)
        # ste_modules[1]: ste module after the batch norm, defining the quantum
        # of the requantized outputs of the bn layer
        self.ste_op = getattr(ste_lib, 'STEActKernel').apply
        self.ste_in = ste_modules[0]
        self.ste_out = ste_modules[1]
        self.start_epoch = start_epoch
        self.num_levels_mult = num_levels_mult
        self.num_levels_add = num_levels_add
        self.register_buffer('step_mult', torch.tensor(step_mult))
        # step_add is given by q_out
        self.started = False
        # all of these are parameters because they are used in calculations ->
        # automatic dtype casting is
        self.register_buffer('q_in', torch.tensor(1.0))
        self.register_buffer('q_out', torch.tensor(1.0))
        self.register_buffer('min_mult_int', torch.tensor(-(num_levels_mult-1)/2))
        self.register_buffer('max_mult_int', torch.tensor((num_levels_mult-1)/2))
        self.register_buffer('min_add_int', torch.tensor(-(num_levels_add-1)/2))
        self.register_buffer('max_add_int', torch.tensor((num_levels_add-1)/2))

    def step(self, epoch):
        if epoch == self.start_epoch:
            # copy the quanta from the linked STE layers
            self.q_in.data = self.ste_in.abs_max_value.data.clone().detach()/(ste_in.num_levels-1)
            self.q_out.data = self.ste_out.abs_max_value.data.clone().detach()/(ste_out.num_levels-1)
        if epoch >= self.start_epoch:
            self.started = True

    def get_gamma_tilde(self):
         # fold gamma, variance and requantization into one
            # multiplicative factor
        if self.track_running_stats:
            sigma = torch.sqrt(self.running_var + self.eps)
        else:
            sigma = torch.ones(self.num_features, dtype=self.weight.data.dtype)
        if self.affine:
            gamma = self.weight
        else:
            gamma = torch.ones(self.num_features, dtype=self.weight.data.dtype)
        gamma_tilde = gamma/sigma*self.q_in/self.q_out
        # quantize the scaled gamma tilde, so it is "aligned" on the
        # quantization grid for the multiplicative factor
        gamma_tilde = self.ste_op(gamma_tilde, self.step_mult, self.min_mult_int, self.max_mult_int)
        # now, scale gamma_tilde back so it doesn't perform requantization
        gamma_tilde = gamma_tilde * self.q_out / self.q_in
        return gamma_tilde

    def get_beta_tilde(self):
        # quantize the effective bias to the quantum defined by the ste_out
        # layer

        if self.track_running_stats:
            sigma = torch.sqrt(self.running_var + self.eps)
        else:
            sigma = torch.ones(self.num_features, dtype=self.weight.data.dtype)

        if self.track_running_stats:
            beta_tilde = -self.running_mean/sigma
        else:
            beta_tilde = torch.zeros(self.num_features, dtype=self.bias.data.dtype)
        if self.affine:
            beta_tilde += self.bias

        beta_tilde = self.ste_op(beta_tilde, self.q_out, self.min_add_int, self.max_add_int)
        return beta_tilde

    def forward(self, x):
        if not self.started:
            return super(_STEBatchNorm, self).forward(x)
        else:
            self._check_input_dim(x)
            gamma_tilde = self.get_gamma_tilde()
            beta_tilde = self.get_beta_tilde()
            return x*gamma_tilde + beta_tilde


class STEBatchNorm1d(_STEBatchNorm, nn.BatchNorm1d):
    def __init__(self, ste_modules, start_epoch, num_levels_mult, num_levels_add, step_mult, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        _STEBatchNorm.__init__(self, ste_modules=ste_modules, start_epoch=start_epoch, num_features=num_features, num_levels_mult=num_levels_mult, num_levels_add=num_levels_add, step_mult=step_mult, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def get_gamma_tilde(self):
        gamma_tilde =  super(STEBatchNorm1d, self).get_gamma_tilde()
        gamma_tilde = gamma_tilde.unsqueeze(0)
        gamma_tilde = gamma_tilde.unsqueeze(2)
        return gamma_tilde

    def get_beta_tilde(self):
        beta_tilde = super(STEBatchNorm1d, self).get_beta_tilde()
        beta_tilde = beta_tilde.unsqueeze(0)
        beta_tilde = beta_tilde.unsqueeze(2)
        return beta_tilde


class STEBatchNorm2d(_STEBatchNorm, nn.BatchNorm2d):
    def __init__(self, ste_modules, start_epoch, num_levels_mult, num_levels_add, step_mult, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        _STEBatchNorm.__init__(self, ste_modules=ste_modules, start_epoch=start_epoch, num_features=num_features, num_levels_mult=num_levels_mult, num_levels_add=num_levels_add, step_mult=step_mult, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def get_gamma_tilde(self):
        gamma_tilde =  super(STEBatchNorm2d, self).get_gamma_tilde()
        gamma_tilde = gamma_tilde.unsqueeze(0)
        gamma_tilde = gamma_tilde.unsqueeze(2)
        gamma_tilde = gamma_tilde.unsqueeze(3)
        return gamma_tilde

    def get_beta_tilde(self):
        beta_tilde = super(STEBatchNorm2d, self).get_beta_tilde()
        beta_tilde = beta_tilde.unsqueeze(0)
        beta_tilde = beta_tilde.unsqueeze(2)
        beta_tilde = beta_tilde.unsqueeze(3)
        return beta_tilde


class STEBatchNorm3d(_STEBatchNorm, nn.BatchNorm3d):
    def __init__(self, ste_modules, start_epoch, num_levels_mult, num_levels_add, step_mult, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        _STEBatchNorm.__init__(self, ste_modules=ste_modules, start_epoch=start_epoch, num_features=num_features, num_levels_mult=num_levels_mult, num_levels_add=num_levels_add, step_mult=step_mult, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def get_gamma_tilde(self):
        gamma_tilde =  super(STEBatchNorm3d, self).get_gamma_tilde()
        gamma_tilde = gamma_tilde.unsqueeze(0)
        gamma_tilde = gamma_tilde.unsqueeze(2)
        gamma_tilde = gamma_tilde.unsqueeze(3)
        gamma_tilde = gamma_tilde.unsqueeze(4)
        return gamma_tilde

    def get_beta_tilde(self):
        beta_tilde = super(STEBatchNorm3d, self).get_beta_tilde()
        beta_tilde = beta_tilde.unsqueeze(0)
        beta_tilde = beta_tilde.unsqueeze(2)
        beta_tilde = beta_tilde.unsqueeze(3)
        beta_tilde = beta_tilde.unsqueeze(4)
        return beta_tilde

#if __name__ == "__main__":
#     u = torch.randn(10, requires_grad=True)
#     x = u * 2
#     y = STEActivation(num_levels=2)(x)
#     L = y.norm(2)  # pull to 0
#     L.backward()
    # true_bn = nn.BatchNorm1d(num_features=16)
    # ste_in = STEActivation(255, 0)
    # ste_in.abs_max_value.data = torch.tensor(2.0)
    # ste_out = STEActivation(255, 0)
    # ste_out.abs_max_value.data = torch.tensor(5.0)
    # q_bn = STEBatchNorm1d([ste_in, ste_out], 1, 2**10, 255, 8/2**10, 16)

    # test_inputs = [torch.rand(4, 16, 20) for k in range(25)]
    # test_outputs = [torch.rand(4, 16, 20) for k in range(25)]
    # opt_true =  torch.optim.SGD(true_bn.parameters(), lr=0.0005, momentum=0.0)
    # opt_q =  torch.optim.SGD(q_bn.parameters(), lr=0.0005, momentum=0.0)
    # quant_loss = nn.MSELoss()
    # true_loss = nn.MSELoss()
    # for i, o in zip(test_inputs, test_outputs):
    #     opt_true.zero_grad()
    #     opt_q.zero_grad()
    #     true_out = true_bn(i)
    #     q_out = q_bn(i)
    #     t_loss = true_loss(true_out, o)
    #     q_loss = quant_loss(q_out, o)
    #     print("True loss: ", t_loss.item())
    #     print("Quant loss: ", q_loss.item())
    #     t_loss.backward()
    #     q_loss.backward()
    #     opt_true.step()
    #     opt_q.step()

    # # after this, the parameters of both should be the same
    # print("abs diff between weights: ", torch.sum(torch.abs(true_bn.weight.data.clone().detach()-q_bn.weight.data.clone())))
    # print("qbn weight: ", q_bn.weight.data)
    # print("tbn weight: ", true_bn.weight.data)
    # print("qbn bias: ", q_bn.bias.data)
    # print("tbn bias: ", true_bn.bias.data)
    # print("abs diff between bias: ", torch.sum(torch.abs(true_bn.bias.data.clone().detach()-q_bn.bias.data.clone())))

    # q_bn.step(1)
    # # now we should see small differences
    # ###true_bn.eval()
    # #q_bn.eval()

    # for i, o in zip(test_inputs, test_outputs):
    #     opt_true.zero_grad()
    #     opt_q.zero_grad()
    #     true_out = true_bn(i)
    #     q_out = q_bn(i)
    #     t_loss = true_loss(true_out, o)
    #     q_loss = quant_loss(q_out, o)
    #     print("True loss: ", t_loss.item())
    #     print("Quant loss: ", q_loss.item())
    #     t_loss.backward()
    #     q_loss.backward()
    #     opt_true.step()
    #     opt_q.step()

    # print("abs diff between weights: ", torch.sum(torch.abs(true_bn.weight.data.clone().detach()-q_bn.weight.data.clone())))
    # print("abs diff between bias: ", torch.sum(torch.abs(true_bn.bias.data.clone().detach()-q_bn.bias.data.clone())))
    # print("qbn running mean: ", q_bn.running_mean.data)
    # print("tbn running mean: ", true_bn.running_mean.data)
    # print("qbn weight: ", q_bn.weight.data)
    # print("tbn weight: ", true_bn.weight.data)
    # print("qbn bias: ", q_bn.bias.data)
    # print("tbn bias: ", true_bn.bias.data)
    # ones_in = torch.ones(2,16, 1)
    # true_bn.eval()
    # print("qbn ones out: ", q_bn(ones_in))
    # print("tbn ones out: ", true_bn(ones_in))
