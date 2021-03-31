import torch
import numpy as np
import horovod.torch as hvd


class Meter(object):
    def __init__(self, pp_pr, pp_gt):
        self.n_tracked    = None
        self.loss         = None
        self.avg_loss     = None
        # main metric is Top1 error
        self.topk         = (1, 5)
        self.correct      = None
        self.avg_error    = None
        self.avg_metric   = None
        self.start_metric = torch.tensor(100.)

        self.pp_pr        = pp_pr
        self.pp_gt        = pp_gt
        self.reset()

    def reset(self):
        self.n_tracked = torch.tensor(0)
        self.loss      = torch.tensor(0.)
        self.avg_loss  = torch.tensor(0.)
        self.correct   = torch.zeros(len(self.topk))
        self.avg_error = self.start_metric.repeat(len(self.topk))

    def update(self, pr_outs, gt_labels, loss):
        pr_labels       = self.pp_pr(pr_outs)
        gt_labels       = self.pp_gt(gt_labels)
        assert len(pr_labels) == len(gt_labels), 'Number of predictions does not match number of ground truths!'
        bs              = torch.tensor(len(gt_labels))
        self.n_tracked += hvd.allreduce(bs, name='batch_size', average=False)
        # update loss
        self.loss      += hvd.allreduce(loss.item() * bs, name='loss', average=False)  # loss should be unlinked from computational graph
        self.avg_loss   = self.loss / self.n_tracked
        # update main metric
        correct = torch.zeros(len(self.topk))
        for i in range(bs):
            for j, k in enumerate(self.topk):
                correct[j] += gt_labels[i] in pr_labels[i][0:k]
        self.correct   += hvd.allreduce(correct, name='metric', average=False)
        self.avg_error  = 100. * (1. - np.true_divide(self.correct, self.n_tracked))  # cast to avoid integer division
        self.avg_metric = self.avg_error[0]  # main metric is Top1 error

    def is_better(self, current_metric, best_metric):
        # compare Top1 errors
        return current_metric < best_metric

'''
#test if it works when I take the ImageNet/meter.py instead
import math
import torch
import horovod.torch as hvd


class Meter(object):
    def __init__(self, pp_pr, pp_gt):
        self.n_tracked    = None
        self.loss         = None
        self.avg_loss     = None
        # main metric is classification error
        self.pp_pr        = pp_pr
        self.pp_gt        = pp_gt
        self.start_metric = 100.
        self.correct      = None
        self.avg_metric   = None
        self.reset()

    def reset(self):
        self.n_tracked  = 0
        self.loss       = 0.
        self.avg_loss   = 0.
        self.correct    = 0
        self.avg_metric = self.start_metric

    def update(self, pr_outs, gt_labels, loss, track_metric=False):
        gt_labels = self.pp_gt(gt_labels)
        batch_size      = len(gt_labels)
        self.n_tracked += batch_size
        # update loss
        self.loss      += loss * batch_size
        self.avg_loss   = self.loss / self.n_tracked
        if track_metric:
            # update main metric
            pr_labels = self.pp_pr(pr_outs)
            assert len(pr_labels) == len(gt_labels), 'Number of predictions and number of ground truths do not match!'
            for i in range(len(pr_labels)):
                self.correct += pr_labels[i] == gt_labels[i]
            self.avg_metric = 100. * (1. - self.correct / self.n_tracked)

    def is_better(self, current_metric, best_metric):
        # compare classification errors
        return current_metric < best_metric

    def bar(self):
        return '| Loss: {loss:8.5f} | Accuracy: {acc:6.2f}%%'.format(
                loss=self.avg_loss,
                acc=100. - self.avg_metric)
    
'''