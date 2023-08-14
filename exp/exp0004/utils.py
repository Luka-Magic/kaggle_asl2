import os
import numpy as np
import torch
import warnings
from typing import List, Dict, Union, Tuple, Any
import math
from Levenshtein import distance


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    warnings.simplefilter('ignore')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def validation_metrics(pred, target):
    '''
        Input:
            pred: list
                bs, seq_len
            target: list
                bs, seq_len
    '''
    # calc acc
    bs = len(pred)

    # calc levenstein distance
    sum_norm_ld = 0
    sum_acc = 0
    for i in range(bs):
        N = len(target[i])
        D = distance(pred[i], target[i])
        sum_norm_ld += (N - D) / N
        sum_acc += int(pred[i] == target[i])
    norm_ld = sum_norm_ld / bs
    acc = sum_acc / bs
    return acc, norm_ld