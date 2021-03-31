
import torch


def postprocess_pr(pr_outs):
    _, topk = torch.topk(pr_outs.detach().cpu(), 5, dim=1)
    return [[a.item() for a in p] for p in topk]


def postprocess_gt(gt_labels):
    return [l.item() for l in gt_labels.detach().cpu()]

'''
#test if it works when I take the ImageNet/VGG/postprocess.py
import torch


def postprocess_pr(pr_outs):
    _, pr_outs = torch.max(pr_outs, dim=1)
    return [p.item() for p in pr_outs.detach().cpu()]


def postprocess_gt(gt_labels):
    return [l.item() for l in gt_labels.detach().cpu()]
'''