import re
import torch
import torch.nn as nn


__all__ = ['VGG', 'vgg_torchvision_to_quantlab']

__CONFIGS__ = {
    'VGG11': ['M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, config, bn=False, num_classes=1000):
        super(VGG, self).__init__()
        if bn:
            adapter = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=not bn),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        else:
            adapter = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        self.adapter = adapter
        self.features = self._make_features(__CONFIGS__[config], bn)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    @staticmethod
    def _make_features(config, bn):
        layers = []
        in_channels = 64
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=not bn)]
                layers += [nn.BatchNorm2d(v)] if bn else []
                layers += [nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.adapter(x)
        x = self.features(x)
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)  # enable conversion to Caffe model; to understand difference see:
                                   # https://stackoverflow.com/questions/57234095/what-is-the-difference-of-flatten-and-view-1-in-pytorch
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def vgg_torchvision_fold_bias_to_bn(state_dict):
    """Folds convolution biases into BatchNorm layers and returns the modified state_dict without the bias keys"""

    def get_conv_bn_pairs(state_dict):
        from collections import OrderedDict
        # returns the key prefixes for all conv+bn pairs in a list of dicts
        feature_re = re.compile('features\.\d+')
        results = map(feature_re.search, state_dict.keys())
        feature_prefixes = [r.group(0) for r in results if r is not None]
        # we will have multiple occurrences of the prefixes, so use a hack to uniquify them
        feature_prefixes = list(OrderedDict.fromkeys(feature_prefixes))
        pairs = [{'conv':feature_prefixes[i], 'bn':feature_prefixes[i+1]} for i in range(0, len(feature_prefixes), 2)]
        return pairs

    def fold_bias_to_bn(state_dict, pair, eps):
        # we no longer want the conv_bias key to be in the dict
        conv_bias = state_dict.pop(pair['conv']+'.bias')
        bn_var = state_dict[pair['bn']+'.running_var']
        sigma = torch.sqrt(bn_var + eps)
        gamma = state_dict[pair['bn']+'.weight']
        bias_correction = gamma * conv_bias / sigma
        # fold the conv bias into the BN
        state_dict[pair['bn']+'.bias'] += bias_correction
        return state_dict

    pairs = get_conv_bn_pairs(state_dict)
    # in the unlikely case that the default 'eps' for BN changes with pyTorch releases, get it from a real BN instance
    dummy_bn = nn.BatchNorm2d(1)
    eps = dummy_bn.eps
    for p in pairs:
        fold_bias_to_bn(state_dict, p, eps)

    return state_dict


def vgg_torchvision_to_quantlab(state_dict_tv, state_dict_ql):
    """Takes a state_dict for the TorchVision VGG and converts it to one compatible with the above-defined VGG class"""
    # this dict takes a TORCHVISION version state_dict key and returns the corresponding QUANTLAB
    # version key
    state_dict_conversion_key = {}

    # TorchVision VGGs use convolutions with biases even when BN is enabled, so these biases need to be folded into the BatchNorm...
    state_dict_tv = vgg_torchvision_fold_bias_to_bn(state_dict_tv)

    # now, the only difference between TV and QL VGGs is the presence of the
    # 'adapter' module
    adapter_keys = [k for k in state_dict_ql.keys() if 'adapter' in k and 'num_batches_tracked' not in k]
    for k in adapter_keys:
        tv_key = k.replace('adapter', 'features')
        state_dict_conversion_key[tv_key] = k

    # 'features' indices are shifted
    idx_re = re.compile('\d+')

    n_adapter_layers = 3

    tv_features_keys = [k for k in state_dict_tv.keys() if 'features' in k and k not in state_dict_conversion_key.keys()]
    for k in tv_features_keys:
        idx = int(idx_re.search(k).group(0))
        new_idx = idx - n_adapter_layers
        ql_k = k.replace(str(idx), str(new_idx))
        state_dict_conversion_key[k] = ql_k

    # the classifier module is identical
    classifier_keys = [k for k in state_dict_tv.keys() if 'classifier' in k]
    for k in classifier_keys:
        state_dict_conversion_key[k] = k

    # convert state_dict
    new_state_dict = {}
    for k in state_dict_conversion_key.keys():
        ql_key = state_dict_conversion_key[k]
        new_state_dict[ql_key] = state_dict_tv[k]

    return new_state_dict
