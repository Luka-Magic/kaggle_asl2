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


class Model(nn.Module):
    def __init__(self, seq_len, n_features, n_class):
        super(Model, self).__init__()
        self.bn0 = nn.BatchNorm1d(n_features)  # 各landmarkでmean,std=0,1にする
        self.conv1 = nn.Conv1d(n_features, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.sequence_models = nn.Sequential(
            BidirectionalLSTM(128, 256, 256),
            BidirectionalLSTM(256, 256, 256),
        )
        self.fc = nn.Linear(256, n_class)

    def forward(self, x):
        '''
            input: (bs, 42, 576)
            output: (bs, 576(seq_len), 59(n_classes))
        '''
        # x = self.bn0(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))  # (bs, 42, 576)
        x = x.permute(0, 2, 1)  # (bs, 576, 42) B T C
        x = self.sequence_models(x)
        x = self.fc(x)  # (bs, 42, 59)
        return x
