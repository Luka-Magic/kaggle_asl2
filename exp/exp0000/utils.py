import os
import numpy as np
import torch
import warnings
from typing import List, Dict, Union, Tuple, Any
import math
from metrics import normalized_rmse
from levenshtein import distance


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
            pred: (bs, seq_len)
            target: (bs, seq_len)
    '''
    # calc acc
    bs = pred.shape[0]
    acc = np.mean((pred == target).astype(np.float))

    # calc levenstein distance
    sum_norm_ld = 0
    for i in range(bs):
        N = len(target[i])
        D = distance(pred[i], target[i])
        sum_norm_ld += (N - D) / N
    norm_ld = sum_norm_ld / bs
    return acc, norm_ld


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, char_to_idx):
        # character (str): set of the possible characters.
        self.dict = {char: idx+1 for char, idx in char_to_idx.item()}
        self.character = ['[CTCblank]'] + list(self.dict.keys())

    def encode(self, text, batch_max_length=256):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 256 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text, torch.IntTensor(length))

    def decode(self, text_index, length):
        '''convert text-index into text-label.
        input:
            text_index (np.array): text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        '''
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                # removing repeated characters and blank.
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts
