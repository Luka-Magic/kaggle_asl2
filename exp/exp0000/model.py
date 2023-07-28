import math
import torch
from torch import nn
from torch.nn import functional as F


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size,
                           bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        recurrent, _ = self.rnn(input)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class LRUnit1D(nn.Module):
    def __init__(self, in_dim, actf=torch.nn.ReLU()):
        super(LRUnit1D, self).__init__()
        self.layer0 = nn.Linear(in_dim, in_dim, bias=False)
        self.bn0 = nn.BatchNorm1d(in_dim)
        self.layer1 = nn.Linear(in_dim, in_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.actf = actf

    def forward(self, x):
        h = self.actf(self.bn0(self.layer0(x)))
        h = self.bn1(self.layer1(h))
        return x + h


class RSUnit1D(nn.Module):
    def __init__(self, in_dim, kernel_size=3, padding=1,
                 padding_mode='zeros', actf=torch.nn.ReLU()):
        super(RSUnit1D, self).__init__()
        self.layer0 = nn.Conv1d(in_dim, in_dim, kernel_size=kernel_size,
                                padding=padding, padding_mode=padding_mode, bias=False)
        self.bn0 = nn.BatchNorm1d(in_dim)
        self.layer1 = nn.Conv1d(in_dim, in_dim, kernel_size=kernel_size,
                                padding=padding, padding_mode=padding_mode, bias=False)
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.actf = actf

    def forward(self, x):
        h = self.actf(self.bn0(self.layer0(x)))
        h = self.bn1(self.layer1(h))
        return x + h


class Model(nn.Module):
    def __init__(self, seq_len, n_features, n_class):
        super(Model, self).__init__()
        self.actf = nn.LeakyReLU(negative_slope=0.1)
        # self.bn0 = nn.BatchNorm1d(n_features)  # 各landmarkでmean,std=0,1にする
        self.conv1 = nn.Conv1d(n_features, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)  # 各landmarkでmean,std=0,1にする
        self.conv2 = RSUnit1D(128, 3, padding=1, actf=self.actf)
        self.conv3 = RSUnit1D(128, 3, padding=1, actf=self.actf)
        self.conv4 = RSUnit1D(128, 3, padding=1, actf=self.actf)
        self.conv5 = nn.Conv1d(
            128, 256, kernel_size=1, padding=0, bias=True)
        self.sequence_models = nn.Sequential(
            BidirectionalLSTM(256, 256, 256),
            BidirectionalLSTM(256, 256, 256),
        )
        self.linear0 = nn.Linear(256, 128)
        self.fc = nn.Linear(128, n_class)

    def forward(self, x):
        '''
            input: (bs, 42, 576)
            output: (bs, 576(seq_len), 59(n_classes))
        '''
        x = self.actf(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)  # (bs, 256, 576)
        x = x.permute(0, 2, 1)  # (bs, 576, 256) B T C
        x = self.sequence_models(x)
        x = self.linear0(x)
        x = self.fc(x)
        return x
