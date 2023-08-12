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


class LabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, char_to_idx, phrase_max_length, pad_token='P', sos_token='S', eos_token='E'):
        # character (str): set of the possible characters.
        self.phrase_max_length = phrase_max_length
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.dict = {char: idx+3
                     for char, idx in char_to_idx.items()}
        self.dict[pad_token] = 0
        self.dict[sos_token] = 1
        self.dict[eos_token] = 2
        self.character = [pad_token, sos_token,
                          eos_token] + list(self.dict.keys())

    def encode(self, text: str, add_sos=False) -> torch.LongTensor:
        # add sos and eos token
        text = list(text) + [self.eos_token]
        if add_sos:
            text = [self.sos_token] + text

        tensor_length = torch.tensor(
            min(len(text), self.phrase_max_length), dtype=torch.int64)

        if len(text) > self.phrase_max_length:
            # truncate
            text = text[:self.phrase_max_length]
        else:
            # padding
            text = text + [self.pad_token] * \
                (self.phrase_max_length - len(text))

        # convert to index
        tokens = torch.LongTensor([self.dict[char] for char in text])
        return tokens, tensor_length, text

    def decode(self, text_index: torch.LongTensor) -> str:
        # convert tensor to list
        text_index = text_index.tolist()
        # slice sos and eos token
        if 1 in text_index:
            text_index = text_index[text_index.index(1)+1:]
        if 2 in text_index:
            text_index = text_index[:text_index.index(2)]
        texts = ''.join([self.character[i]
                        for i in text_index]).replace(self.pad_token, '')
        return texts
