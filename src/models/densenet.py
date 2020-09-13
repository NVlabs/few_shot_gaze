#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello.
# --------------------------------------------------------

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DenseNet(nn.Module):

    p_dropout = 0.0  # DON'T use this

    num_blocks = 4
    num_layers_per_block = 4
    use_bottleneck = False  # Enabling this usually makes training unstable
    compression_factor = 1.0  # Makes training less stable if != 1.0

    fc_feats = [2]

    def __init__(self, growth_rate=8, activation_fn=nn.ReLU,
                 normalization_fn=nn.BatchNorm2d):
        super(DenseNet, self).__init__()

        # Initial down-sampling conv layers
        self.initial = DenseNetInitialLayers(growth_rate=growth_rate,
                                             activation_fn=activation_fn,
                                             normalization_fn=normalization_fn)
        c_now = self.initial.c_now

        assert (self.num_layers_per_block % 2) == 0
        for i in range(self.num_blocks):
            i_ = i + 1
            # Define dense block
            self.add_module('block%d' % i_, DenseNetBlock(
                c_now,
                num_layers=(int(self.num_layers_per_block / 2)
                            if self.use_bottleneck
                            else self.num_layers_per_block),
                growth_rate=growth_rate,
                p_dropout=self.p_dropout,
                activation_fn=activation_fn,
                normalization_fn=normalization_fn,
                use_bottleneck=self.use_bottleneck,
            ))
            c_now = list(self.children())[-1].c_now

            # Define transition block if not last layer
            if i < (self.num_blocks - 1):
                self.add_module('trans%d' % i_, DenseNetTransitionDown(
                    c_now, p_dropout=self.p_dropout,
                    compression_factor=self.compression_factor,
                    activation_fn=activation_fn,
                    normalization_fn=normalization_fn,
                ))
                c_now = list(self.children())[-1].c_now

        # Final FC layers
        self.fcs = []
        f_now = c_now
        for f in self.fc_feats:
            fc = nn.Linear(f_now, f).to(device)
            fc.weight.data.normal_(0, 0.01)
            fc.bias.data.fill_(0)
            self.fcs.append(fc)
            f_now = f
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, x):
        # Apply initial layers and dense blocks
        for name, module in self.named_children():
            if name == 'fcs':
                break
            x = module(x)

        # Global average pooling
        x = torch.mean(x, dim=2)  # reduce h
        x = torch.mean(x, dim=2)  # reduce w

        # fc to gaze direction
        for fc in self.fcs:
            x = fc(x)

        return x


class DenseNetInitialLayers(nn.Module):

    def __init__(self, growth_rate=8, activation_fn=nn.ReLU,
                 normalization_fn=nn.BatchNorm2d):
        super(DenseNetInitialLayers, self).__init__()
        c_next = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, c_next, bias=False,
                               kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight.data)

        self.norm = normalization_fn(c_next, track_running_stats=False).to(device)
        self.act = activation_fn(inplace=True)

        c_out = 4 * growth_rate
        self.conv2 = nn.Conv2d(2 * growth_rate, c_out, bias=False,
                               kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight.data)

        self.c_now = c_out
        self.c_list = [c_next, c_out]

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        prev_scale_x = x
        x = self.conv2(x)
        return x, prev_scale_x


class DenseNetBlock(nn.Module):

    def __init__(self, c_in, num_layers=4, growth_rate=8, p_dropout=0.1,
                 use_bottleneck=False, activation_fn=nn.ReLU,
                 normalization_fn=nn.BatchNorm2d, transposed=False):
        super(DenseNetBlock, self).__init__()
        self.use_bottleneck = use_bottleneck
        c_now = c_in
        for i in range(num_layers):
            i_ = i + 1
            if use_bottleneck:
                self.add_module('bneck%d' % i_, DenseNetCompositeLayer(
                    c_now, 4 * growth_rate, kernel_size=1, p_dropout=p_dropout,
                    activation_fn=activation_fn,
                    normalization_fn=normalization_fn,
                ))
            self.add_module('compo%d' % i_, DenseNetCompositeLayer(
                4 * growth_rate if use_bottleneck else c_now, growth_rate,
                kernel_size=3, p_dropout=p_dropout,
                activation_fn=activation_fn,
                normalization_fn=normalization_fn,
                transposed=transposed,
            ))
            c_now += list(self.children())[-1].c_now
        self.c_now = c_now

    def forward(self, x):
        x_before = x
        for i, (name, module) in enumerate(self.named_children()):
            if ((self.use_bottleneck and name.startswith('bneck'))
                    or name.startswith('compo')):
                x_before = x
            x = module(x)
            if name.startswith('compo'):
                x = torch.cat([x_before, x], dim=1)
        return x


class DenseNetTransitionDown(nn.Module):

    def __init__(self, c_in, compression_factor=0.1, p_dropout=0.1,
                 activation_fn=nn.ReLU, normalization_fn=nn.BatchNorm2d):
        super(DenseNetTransitionDown, self).__init__()
        c_out = int(compression_factor * c_in)
        self.composite = DenseNetCompositeLayer(
            c_in, c_out,
            kernel_size=1, p_dropout=p_dropout,
            activation_fn=activation_fn,
            normalization_fn=normalization_fn,
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c_now = c_out

    def forward(self, x):
        x = self.composite(x)
        x = self.pool(x)
        return x


class DenseNetCompositeLayer(nn.Module):

    def __init__(self, c_in, c_out, kernel_size=3, growth_rate=8, p_dropout=0.1,
                 activation_fn=nn.ReLU, normalization_fn=nn.BatchNorm2d,
                 transposed=False):
        super(DenseNetCompositeLayer, self).__init__()
        self.norm = normalization_fn(c_in, track_running_stats=False).to(device)
        self.act = activation_fn(inplace=True)
        if transposed:
            assert kernel_size > 1
            self.conv = nn.ConvTranspose2d(c_in, c_out, kernel_size=kernel_size,
                                           padding=1 if kernel_size > 1 else 0,
                                           stride=1, bias=False).to(device)
        else:
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1,
                                  padding=1 if kernel_size > 1 else 0, bias=False).to(device)
        nn.init.kaiming_normal_(self.conv.weight.data)
        self.drop = nn.Dropout2d(p=p_dropout) if p_dropout > 1e-5 else None
        self.c_now = c_out

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        if self.drop is not None:
            x = self.drop(x)
        return x
