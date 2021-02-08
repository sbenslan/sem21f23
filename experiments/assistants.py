import os
import requests
from requests.exceptions import MissingSchema
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torch.optim as optim

import horovod.torch as hvd

import utils.lr_schedulers as lr_schedulers


def get_data(logbook):
    """Return data for the experiment."""

    # create data sets
    train_set, valid_set = logbook.lib.load_data_sets(logbook)
    # is cross-validation experiment?
    if logbook.config['experiment']['n_folds'] > 1:
        import itertools
        torch.manual_seed(logbook.config['experiment']['seed'])  # make data set random split consistent
        indices = torch.randperm(len(train_set)).tolist()
        folds_indices = []
        for k in range(logbook.config['experiment']['n_folds']):
            folds_indices.append(indices[k::logbook.config['experiment']['n_folds']])
        train_fold_indices = list(itertools.chain(*[folds_indices[i] for i in range(len(folds_indices)) if i != logbook.i_fold]))
        valid_fold_indices = folds_indices[logbook.i_fold]
        valid_set = torch.utils.data.Subset(train_set, valid_fold_indices)
        train_set = torch.utils.data.Subset(train_set, train_fold_indices)  # overwriting `train_set` must be done in right order!

    # create samplers (maybe distributed)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=logbook.sw_cfg['global_size'], rank=logbook.sw_cfg['global_rank'])
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_set, num_replicas=logbook.sw_cfg['global_size'], rank=logbook.sw_cfg['global_rank'])

    # wrap data sets into loaders
    bs_train = logbook.config['data']['bs_train']
    bs_valid = logbook.config['data']['bs_valid']
    kwargs = {'num_workers': logbook.hw_cfg['n_cpus'] // logbook.sw_cfg['local_size'], 'pin_memory': True} if logbook.hw_cfg['n_gpus'] else {'num_workers': 1}
    if hasattr(train_set, 'collate_fn'):  # if one data set needs `collate`, all the data sets should need it
        train_l = torch.utils.data.DataLoader(train_set, batch_size=bs_train, sampler=train_sampler, collate_fn=train_set.collate_fn, **kwargs)
        valid_l = torch.utils.data.DataLoader(valid_set, batch_size=bs_valid, sampler=valid_sampler, collate_fn=valid_set.collate_fn, **kwargs)
    else:
        train_l = torch.utils.data.DataLoader(train_set, batch_size=bs_train, sampler=train_sampler, **kwargs)
        valid_l = torch.utils.data.DataLoader(valid_set, batch_size=bs_valid, sampler=valid_sampler, **kwargs)

    return train_l, valid_l


def get_network(logbook):
    """Return a network for the experiment and the loss function for training."""

    def get_pretrained_model_path(url_or_path):
        """If given URL, will try to download the file at that URL if it does not exist
        already and return the path to the file. If given a path, will return
        that path or raise FileNotFound. If the argument is simply a filename
        without preceding path, the file is assumed to be located in the
        '<problem>/<topology>/pretrained' folder.

        """
        #create a "pretrained" folder if it doesn't exist yet
        pretrain_dir = os.path.join(logbook.dir_topology, 'pretrained')
        if not os.path.isdir(pretrain_dir):
            os.mkdir(pretrain_dir)
        try:
            pretrained_url = urlparse(url_or_path)
            model_path = os.path.join(pretrain_dir, os.path.basename(pretrained_url.path))
            if os.path.exists(model_path):
                return model_path
            else:
                r = requests.get(url_or_path, stream=True)
                print("Downloading pretrained model from '{}'...".format(url_or_path))
                open(model_path, 'wb').write(r.content)
                print("Done!")
                return model_path
        except MissingSchema:
            # this means it wasn't a proper URL, so interpret it as a path
            if os.path.dirname(url_or_path) == '':
                # no directory -> should be located in the 'pretrained' dir
                model_path = os.path.join(pretrain_dir, url_or_path)
            else:
                model_path = url_or_path
            if not os.path.exists(model_path):
                raise FileNotFoundError("Pretrained model not found at '{}'!".format(os.path.abspath(model_path)))
            return model_path

    def convert_state_dict(fn_name, state_dict_pt, state_dict_ql):
        """The key 'fn_name' is expected to be in logbook.config['network'] and should
        specify a member of the 'topology' module which takes the state_dict of
        the pretrained model and that of the equivalent QuantLab model. This
        function should convert the pretrained state_dict to one compatible
        with the QuantLab model.
        """

        # if the fn_name key does not exist in the logbook config
        try:
            sd_convert_fn = getattr(logbook.lib, logbook.config['network'][fn_name])
        except KeyError:
            return state_dict_pt

        state_dict_converted = state_dict_pt

        if sd_convert_fn is not None:
            state_dict_converted = sd_convert_fn(state_dict_pt, state_dict_ql)

        return state_dict_converted

    # create the network
    net = getattr(logbook.lib, logbook.config['network']['class'])(**logbook.config['network']['params'])

    # if specified, load pretrained checkpoint for unquantized network
    load_unq_pretrained = False
    try:
        model_file = logbook.config['network']['pretrained_unquantized']
    except KeyError:
        model_file = None
    if model_file is not None:
        model_path = get_pretrained_model_path(model_file)
        state_dict_pt = torch.load(model_path)
        # if required, convert the state_dict
        state_dict_pt = convert_state_dict('state_dict_conversion_unquantized', state_dict_pt, net.state_dict())
        net.load_state_dict(state_dict_pt)
        load_unq_pretrained = True


    # quantize (if specified)
    if logbook.config['network']['quantize'] is not None:
        quant_convert = getattr(logbook.lib, logbook.config['network']['quantize']['routine'])
        net = quant_convert(logbook.config['network']['quantize'], net)

    # if specified, load pretrained checkpoint for quantized network
    try:
        model_file = logbook.config['network']['pretrained_quantized']
    except KeyError:
        model_file = None
    if model_file is not None:
        if load_unq_pretrained:
            print("Warning: Loading of unquantized pretrained weights has no effect - quantized pretrained model is specified as well!")
            model_path = get_pretrained_model_path(model_file)
            state_dict_pt = torch.load(model_path)
            # if required, convert the state_dict
            state_dict_pt = convert_state_dict('state_dict_conversion_quantized', state_dict_pt, net.state_dict())
            net.load_state_dict(state_dict_pt)

    # move to proper device
    net = net.to(logbook.hw_cfg['device'])

    return net


def get_training(logbook, net):
    """Return a training procedure for the experiment."""

    # loss function
    loss_fn_choice = {**nn.__dict__, **logbook.lib.__dict__}
    loss_fn_class  = loss_fn_choice[logbook.config['training']['loss_function']['class']]
    if 'net' in loss_fn_class.__init__.__code__.co_varnames:
        loss_fn = loss_fn_class(net, **logbook.config['training']['loss_function']['params'])
    else:
        loss_fn = loss_fn_class(**logbook.config['training']['loss_function']['params'])

    # optimization algorithm
    opt_choice = {**optim.__dict__}
    logbook.config['training']['optimizer']['params']['lr'] *= logbook.sw_cfg['global_size']  # adjust learning rate
    opt        = opt_choice[logbook.config['training']['optimizer']['class']](net.parameters(), **logbook.config['training']['optimizer']['params'])
    opt        = hvd.DistributedOptimizer(opt, named_parameters=net.named_parameters())

    # learning rate scheduler
    lr_sched_choice = {**optim.lr_scheduler.__dict__, **lr_schedulers.__dict__}
    lr_sched        = lr_sched_choice[logbook.config['training']['lr_scheduler']['class']](opt, **logbook.config['training']['lr_scheduler']['params'])

    # quantization controllers (if specified)
    if logbook.config['training']['quantize']:
        quant_controls = getattr(logbook.lib, logbook.config['training']['quantize']['routine'])
        ctrls = quant_controls(logbook.config['training']['quantize'], net)
    else:
        ctrls = []

    return loss_fn, opt, lr_sched, ctrls
