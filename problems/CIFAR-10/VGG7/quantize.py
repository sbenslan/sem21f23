from torch import nn
import quantlib.graphs as qg
import quantlib.algorithms as qa
from quantlib.graphs.edit import _replace_node
from quantlib.algorithms.ste import STEActivation, STEBatchNorm2d

# adapted from ImageNet VGG quantize.py

__all__ = ['features_ste_inq', 'features_ste_inq_get_controllers']


def features_ste_inq(config, net):

    def get_features_conv_nodes(net):

        net_nodes = qg.analyse.list_nodes(net, verbose=False)

        rule1 = qg.analyse.get_rules_multiple_blocks(['features'])
        rule2 = qg.analyse.rule_linear_nodes
        conv_nodes = qg.analyse.find_nodes(net_nodes, rule1+[rule2], mix='and')
        # don't quantize the first conv layer
        conv_nodes = conv_nodes[1:]
        return conv_nodes

    def get_quant_sequences(net):
        """Find STE-Conv-BN-Act-(Pool-)STE sequences"""
        nodes = qg.list_nodes(net, verbose=False)
        rule1 = qg.analyse.get_rules_multiple_blocks(['features'])
        features_nodes = qg.analyse.find_nodes(nodes, rule1, mix='or')
        sequences = []
        for i in range(len(features_nodes)):
            try:
                if (isinstance(features_nodes[i].module, STEActivation) and
                   isinstance(features_nodes[i+1].module, nn.Conv2d) and
                   isinstance(features_nodes[i+2].module, nn.BatchNorm2d) and
                   isinstance(features_nodes[i+3].module, nn.ReLU)):
                    if isinstance(features_nodes[i+4].module, nn.MaxPool2d) and isinstance(features_nodes[i+5].module, STEActivation):
                        l = 6
                    elif isinstance(features_nodes[i+4].module, STEActivation):
                        l = 5
                    else:
                        continue
                    sequences.append(features_nodes[i:i+l])
            except IndexError:
                pass
        return sequences

    # add STE in front of convolutions
    ste_config = config['STE']
    conv_nodes = get_features_conv_nodes(net)
    qg.edit.add_before_linear_ste(net, conv_nodes, num_levels=ste_config['n_levels'], quant_start_epoch=ste_config['quant_start_epoch'])
    # add a last STE after the last 'features' layer
    f_rule = qg.analyse.get_rules_multiple_blocks(['features'])
    features_nodes = qg.find_nodes(qg.list_nodes(net), f_rule, mix='or')

    last_f_node = features_nodes[-1]
    m = last_f_node.module
    last_ste_m = nn.Sequential(m, STEActivation(num_levels=ste_config['n_levels'], quant_start_epoch=ste_config['quant_start_epoch']))
    _replace_node(net, last_f_node.name, last_ste_m)

    try:
        if ste_config['bn']:
            n_lvl_add = ste_config['n_levels_bn_add']
            n_lvl_mult = ste_config['n_levels_bn_mult']
            step_mult = ste_config['step_bn_mult']
            seqs = get_quant_sequences(net)
            for seq in seqs:
                ste0 = seq[0]
                bn_node = seq[2]
                ste1 = seq[-1]
                qbn = STEBatchNorm2d.from_bn((ste0, ste1), start_epoch=ste_config['quant_start_epoch'],
                                             num_levels_mult=n_lvl_mult, num_levels_add=n_lvl_add,
                                             step_mult=step_mult, bn_inst=bn_node.module)
                _replace_node(net, bn_node.name, qbn)
    except KeyError:
        print("No/incorrect BatchNorm quantization specification found; leaving BatchNorm layers unquantized")

    # replace convolutions with INQ counterparts
    inq_config = config['INQ']
    # need to call this again because the STE modification steps changed the names!
    conv_nodes = get_features_conv_nodes(net)
    qg.edit.replace_linear_inq(net, conv_nodes, num_levels=inq_config['n_levels'], quant_init_method=inq_config['quant_init_method'], quant_strategy=inq_config['quant_strategy'])

    return net


def features_ste_inq_get_controllers(config, net):

    net_nodes = qg.analyse.list_nodes(net)
    rule = qg.analyse.get_rules_multiple_blocks(['features'])
    features_nodes = qg.analyse.find_nodes(net_nodes, rule, mix='or')

    # get STE controller
    ste_ctrl_config = config['STE']
    ste_modules = qa.ste.STEController.get_ste_modules(features_nodes)
    ste_controller = qa.ste.STEController(ste_modules, ste_ctrl_config['clear_optim_state_on_step'])

    # get INQ controller
    inq_ctrl_config = config['INQ']
    inq_modules = qa.inq.INQController.get_inq_modules(features_nodes)
    inq_controller = qa.inq.INQController(inq_modules, inq_ctrl_config['schedule'], inq_ctrl_config['clear_optim_state_on_step'])

    return [ste_controller, inq_controller]
