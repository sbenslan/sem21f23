import torch
import torch.nn as nn

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


# if __name__ == "__main__":
#     u = torch.randn(10, requires_grad=True)
#     x = u * 2
#     y = STEActivation(num_levels=2)(x)
#     L = y.norm(2)  # pull to 0
#     L.backward()
