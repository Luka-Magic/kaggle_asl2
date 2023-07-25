import math
import torch
from torch import nn
from torchvision.ops import StochasticDepth


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        cosine = torch.nn.functional.linear(torch.nn.functional.normalize(
            x), torch.nn.functional.normalize(self.weight)).float()
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        if label is None:
            output = cosine
        else:
            one_hot = torch.zeros(cosine.size(), device='cuda')
            one_hot.scatter_(1, label.cuda().view(-1, 1).long(), 1)
            # you can use torch.where if your torch.__version__ is 0.4
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class LRUnit1D(nn.Module):
    def __init__(self, in_dim, stochastic_depth_prob=0.0, actf=torch.nn.ReLU()):
        super(LRUnit1D, self).__init__()
        self.layer0 = nn.Linear(in_dim, in_dim, bias=False)
        self.bn0 = nn.BatchNorm1d(in_dim)
        self.layer1 = nn.Linear(in_dim, in_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.actf = actf
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, x):
        h = self.actf(self.bn0(self.layer0(x)))
        h = self.bn1(self.layer1(h))
        h = self.stochastic_depth(h)
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


class BackboneFixedLen(nn.Module):
    def __init__(self,
                 in_dim=42,
                 dim1=128,
                 dim2=512,
                 kernel_size=3, padding=1,
                 negative_slope=0.1,
                 ):
        super(BackboneFixedLen, self).__init__()
        self.actf = torch.nn.LeakyReLU(negative_slope=negative_slope)
        self.bn0 = nn.BatchNorm1d(in_dim)
        self.conv1 = nn.Conv1d(
            in_dim, dim1, kernel_size=3, padding=1, bias=True)
        self.conv2 = RSUnit1D(dim1, kernel_size=kernel_size,
                              padding=padding, actf=self.actf)
        self.conv3 = RSUnit1D(dim1, kernel_size=kernel_size,
                              padding=padding, actf=self.actf)
        self.conv4 = RSUnit1D(dim1, kernel_size=kernel_size,
                              padding=padding, actf=self.actf)
        self.conv5 = RSUnit1D(dim1, kernel_size=kernel_size,
                              padding=padding, actf=self.actf)
        self.conv6 = RSUnit1D(dim1, kernel_size=kernel_size,
                              padding=padding, actf=self.actf)
        self.conv6a = nn.Conv1d(
            dim1, dim2, kernel_size=1, padding=0, bias=True)

        self.flatten = torch.nn.Flatten(1)
        self.g_pool = torch.nn.AdaptiveMaxPool1d(1, return_indices=False)
        self.pool = torch.nn.MaxPool1d(2)

    def forward(self, x):
        h = self.bn0(x)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.pool(h)
        h = self.conv3(h)
        h = self.pool(h)
        h = self.conv4(h)
        h = self.pool(h)
        h = self.conv5(h)
        h = self.conv6(h)
        h = self.conv6a(h)
        h = self.g_pool(h)
        h = self.flatten(h)
        return h


class ModelFixedLen(nn.Module):
    def __init__(self,
                 in_dim_H=42,
                 in_dim_L=80,
                 dim1_H=128,
                 dim1_L=64,
                 dim2=512,
                 n_class=250,
                 kernel_size=3, padding=1,
                 arc_m=0.5,
                 arc_s=15,
                 easy_margin=False,
                 stochastic_depth_prob=0.0,
                 negative_slope=0.1,
                 n_recyile=1
                 ):
        super(ModelFixedLen, self).__init__()
        self.actf = torch.nn.LeakyReLU(negative_slope=negative_slope)
        self.hand_module = BackboneFixedLen(
            in_dim=in_dim_H, dim1=dim1_H, dim2=dim2, negative_slope=negative_slope)
        self.other_module = BackboneFixedLen(
            in_dim=in_dim_L, dim1=dim1_L, dim2=dim2, negative_slope=negative_slope)

        self.line1 = LRUnit1D(dim2, stochastic_depth_prob, actf=self.actf)
        self.line2 = LRUnit1D(dim2, stochastic_depth_prob, actf=self.actf)
        self.line3 = LRUnit1D(dim2, stochastic_depth_prob, actf=self.actf)
        self.end = ArcMarginProduct(
            dim2, n_class, easy_margin=easy_margin, s=arc_s, m=arc_m)
        self.flatten = torch.nn.Flatten(1)
        self.g_pool = torch.nn.AdaptiveMaxPool1d(1, return_indices=False)
        self.pool = torch.nn.MaxPool1d(2)

        self.n_recyile = n_recyile

    def forward(self, hand, other):
        # (bs, 42, 543*2) (bs, 80, 543*2)
        hand = self.hand_module(hand)
        other = self.other_module(other)
        h = hand + other
        for n in range(self.n_recyile):
            h = self.line1(h)
        for n in range(self.n_recyile):
            h = self.line2(h)
        for n in range(self.n_recyile):
            h = self.line3(h)
        h = self.end(h)
        return h
