import torch
import matplotlib.pyplot as plt


def generate_stair_function(qmin=-1, qmax=1, n_bits=2):

    assert n_bits > 0
    n_levels = 2 ** n_bits - 1 if n_bits > 1 else 2 ** n_bits  # symmetric stair_function
    n_jumps = n_levels - 1

    q = torch.linspace(qmin, qmax, n_levels)

    delta_t = qmax - qmin
    step_t = delta_t / n_jumps
    t = (torch.linspace(qmin, qmax, n_levels) + (step_t / 2))[:-1]

    return t, q


def generate_inputs(N, q, distribution='Dirac', n_iters=10, is_cuda=False):
    """Generate inputs for the specific to the function to be tested.

    Args:
        N: 'int'; the size of the input (in algorithmic sense).
        distribution: 'string'; the probability distribution according to
            which samples are generated; it must be one between "Dirac" (a
            single input is hard-coded in this function and will be replicated
            at measurement time), "random_Dirac" (generate one random input
            which will be replicated at measurement time) and "random"
            (generate 'n_iters' random inputs).
        n_iters: 'int'; the number of inputs to be generated.

    Returns:
        forward_inputs: 'tuple' or 'list'; the inputs on which to measure
            performance of the "forward" methods.
        backward_inputs: 'tuple' or 'list'; the inputs on which to measure
            performance of the "backward" methods.

    """

    #####################################
    ## ANA parameters are defined here ##
    #####################################
    # parameters of the stair function
    qmin = torch.min(q)
    qmax = torch.max(q)
    assert (qmin < qmax)  # function must be increasing
    #####################################

    if distribution == 'Dirac':
        delta_q = qmax - qmin
        bepsx = (10 / 100) * delta_q / 2  # show margins for 5% out of the range of the quantized steps
        xmin = qmin - bepsx
        xmax = qmax + bepsx
        x_in = torch.linspace(xmin, xmax, N)
        grad_in = torch.ones_like(x_in)
        if is_cuda:
            x_in = x_in.cuda()
            grad_in = grad_in.cuda()
        forward_inputs = x_in
        backward_inputs = grad_in

    return forward_inputs, backward_inputs


def measure(anamod, forward_inputs, backward_inputs, n_iters=10):
    """

    Args:
        mod: 'torch.nn.Module'; the module to be profiled.
        inputs: 'tuple' or 'list' of 'tuple's; each tuple must be collection
            of objects matching the signature of the target function.
        n_iters: 'int', if 'inputs' is not a 'list', then it will be fed at
            least 'n_iters' time to get a statistically relevant sample.

    Returns:
        measurements: 'list' of 'float's; one measurement for each input fed.

    """
    import time

    if not isinstance(forward_inputs, list):
        forward_inputs = [forward_inputs] * n_iters
    if not isinstance(backward_inputs, list):
        backward_inputs = [backward_inputs] * n_iters

    forward_measurements = list()
    backward_measurements = list()
    for _input_for, _input_back in zip(forward_inputs, backward_inputs):
        _input_for.requires_grad = True
        # forward
        tic = time.perf_counter()
        x_out = anamod(_input_for)
        toc = time.perf_counter()
        forward_measurements.append((toc - tic) * 10**6)  # in microseconds
        # backward
        tic = time.perf_counter()
        x_out.backward(_input_back)
        toc = time.perf_counter()
        backward_measurements.append((toc - tic) * 10**6)  # in microseconds

    return forward_measurements, backward_measurements


def profile(anamod, N=tuple([10**i for i in range(3, 6)]), n_iters=10):
    import numpy as np

    print("Performance report for {} version".format('GPU' if anamod.stddev.is_cuda else 'CPU'))
    print("Input size \t Forward \t\t\t\t\t\t | Backward")

    for input_size in N:
        forward_inputs, backward_inputs = generate_inputs(input_size, anamod.quant_levels.data, distribution='Dirac', n_iters=n_iters, is_cuda=anamod.stddev.data.is_cuda)

        forward_perf, backward_perf = measure(anamod, forward_inputs, backward_inputs, n_iters=n_iters)

        forward_mu = np.mean(forward_perf)
        forward_std = np.std(forward_perf)

        backward_mu = np.mean(backward_perf)
        backward_std = np.std(backward_perf)

        print("{:10d} \t {:10.3f}us +/- {:10.3f}us \t | {:10.3f}us +/- {:10.3f}us".format(input_size, forward_mu, forward_std, backward_mu, backward_std))


def check_functional_equivalence(anamod_gpu, anamod_cpu, tolerance=1e-6):
    # zero 'tolerance' means to check for EXACT equivalence: same input is mapped to same outputs
    # otherwise, we are satisfied by some APPROXIMATE equivalence between the implementations

    print("Functional equivalence check - Admitted tolerance: {}".format(tolerance))

    forward_inputs_gpu, backward_inputs_gpu = generate_inputs(10**4, anamod_gpu.quant_levels.data, distribution='Dirac', is_cuda=anamod_gpu.stddev.is_cuda)
    forward_inputs_cpu, backward_inputs_cpu = generate_inputs(10**4, anamod_cpu.quant_levels.data, distribution='Dirac', is_cuda=anamod_cpu.stddev.is_cuda)

    forward_inputs_gpu.requires_grad = True
    x_in_gpu = forward_inputs_gpu
    x_out_gpu = anamod_gpu(forward_inputs_gpu)
    x_out_gpu.backward(backward_inputs_gpu)

    forward_inputs_cpu.requires_grad = True
    x_in_cpu = forward_inputs_cpu
    x_out_cpu = anamod_cpu(forward_inputs_cpu)
    x_out_cpu.backward(backward_inputs_cpu)

    forward_error = torch.max(torch.abs(x_out_gpu.cpu() - x_out_cpu.cpu()))
    print("Maximum discrepancy in forward pass: {}".format(forward_error))

    backward_error = torch.max(torch.abs(x_in_gpu.grad.cpu() - x_in_cpu.grad.cpu()))
    print("Maximum discrepancy in backward pass: {}".format(backward_error))

    passed = False
    if (forward_error < tolerance) and (backward_error < tolerance):
        print("Test passed!")
        passed = True
    else:
        print("Test failed!")

    return passed


def show(anamod):
    """Show the graphics of an activation function and its derivative."""

    _input_for, _input_back = generate_inputs(10**4, anamod.quant_levels.data, is_cuda=anamod.stddev.data.is_cuda)

    _input_for.requires_grad = True
    x_in = _input_for
    x_out = anamod(_input_for)
    x_out.backward(_input_back)

    ps = 1  # point size
    fig, (axforward, axbackward) = plt.subplots(1, 2)
    axforward.scatter(x_in.cpu().detach().numpy(), x_out.cpu().detach().numpy(), c='b', s=ps)
    axbackward.scatter(x_in.cpu().detach().numpy(), x_in.grad.cpu().detach().numpy(), c='r', s=ps)


def get_ana_module(module_dict, noise_type, t, q, s, cuda=False):

    anamod = module_dict['ANAActivation'](noise_type, t, q)
    anamod.set_stddev(s)
    if cuda:
        anamod.cuda()

    return anamod


def test_noise(module_dict, noise_type, t, q, s):

    anamod_cpu = get_ana_module(module_dict, noise_type, t, q, s)
    anamod_gpu = get_ana_module(module_dict, noise_type, t, q, s, cuda=True)

    show(anamod_cpu)
    show(anamod_gpu)

    assert check_functional_equivalence(anamod_gpu, anamod_cpu, tolerance=1e-4)

    profile(anamod_cpu)
    profile(anamod_gpu)


def test(module_dict, qmin=-127., qmax=127., n_bits=8, stddev_for=1.0, stddev_back=1.0):

    t, q = generate_stair_function(qmin=qmin, qmax=qmax, n_bits=n_bits)
    s = torch.Tensor([stddev_for, stddev_back])

    for noise_type in ['uniform', 'triangular', 'normal', 'logistic']:
        print("Noise type: {}".format(noise_type.upper()))
        test_noise(module_dict, noise_type, t, q, s)
        print("\n")
