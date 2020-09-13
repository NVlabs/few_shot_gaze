#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello.
# --------------------------------------------------------

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .densenet import (
    DenseNetInitialLayers,
    DenseNetBlock,
    DenseNetTransitionDown,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DTED(nn.Module):

    def __init__(self, z_dim_app, z_dim_gaze, z_dim_head,
                 growth_rate=32, activation_fn=nn.LeakyReLU,
                 normalization_fn=nn.InstanceNorm2d,
                 decoder_input_c=16,
                 normalize_3d_codes=False,
                 normalize_3d_codes_axis=None,
                 use_triplet=False,
                 gaze_hidden_layer_neurons=64,
                 backprop_gaze_to_encoder=False,
                 ):
        super(DTED, self).__init__()

        # Cache some specific configurations
        self.normalize_3d_codes = normalize_3d_codes
        self.normalize_3d_codes_axis = normalize_3d_codes_axis
        self.use_triplet = use_triplet
        self.gaze_hidden_layer_neurons = gaze_hidden_layer_neurons
        self.backprop_gaze_to_encoder = backprop_gaze_to_encoder
        if self.normalize_3d_codes:
            assert self.normalize_3d_codes_axis is not None

        # Define feature map dimensions at bottleneck
        bottleneck_shape = (2, 8)
        self.bottleneck_shape = bottleneck_shape

        # The meaty parts
        self.encoder = DenseNetEncoder(
            num_blocks=4,
            growth_rate=growth_rate,
            activation_fn=activation_fn,
            normalization_fn=normalization_fn,
        )
        c_now = list(self.children())[-1].c_now
        self.decoder_input_c = decoder_input_c
        enc_num_all = np.prod(bottleneck_shape) * self.decoder_input_c
        self.decoder = DenseNetDecoder(
            self.decoder_input_c,
            num_blocks=4,
            growth_rate=growth_rate,
            activation_fn=activation_fn,
            normalization_fn=normalization_fn,
            compression_factor=1.0,
        )

        # The latent code parts
        self.z_dim_app = z_dim_app
        self.z_dim_gaze = z_dim_gaze
        self.z_dim_head = z_dim_head
        z_num_all = 3 * (z_dim_gaze + z_dim_head) + z_dim_app

        self.fc_enc = self.linear(c_now, z_num_all)
        self.fc_dec = self.linear(z_num_all, enc_num_all)

        self.build_gaze_layers(3 * z_dim_gaze)

    def build_gaze_layers(self, num_input_neurons, num_hidden_neurons=64):
        self.gaze1 = self.linear(num_input_neurons, self.gaze_hidden_layer_neurons)
        self.gaze2 = self.linear(self.gaze_hidden_layer_neurons, 3)

    def linear(self, f_in, f_out):
        fc = nn.Linear(f_in, f_out)
        nn.init.kaiming_normal(fc.weight.data)
        nn.init.constant(fc.bias.data, val=0)
        return fc

    def rotate_code(self, data, code, mode, fr=None, to=None):
        """Must calculate transposed rotation matrices to be able to
           post-multiply to 3D codes."""
        key_stem = 'R_' + mode
        if fr is not None and to is not None:
            rotate_mat = torch.matmul(
                data[key_stem + '_' + fr],
                torch.transpose(data[key_stem + '_' + to], 1, 2)
            )
        elif to is not None:
            rotate_mat = torch.transpose(data[key_stem + '_' + to], 1, 2)
        elif fr is not None:
            # transpose-of-inverse is itself
            rotate_mat = data[key_stem + '_' + fr]
        return torch.matmul(code, rotate_mat)

    def encode_to_z(self, data, suffix):
        x = self.encoder(data['image_' + suffix])
        enc_output_shape = x.shape
        x = x.mean(-1).mean(-1)  # Global-Average Pooling

        # Create latent codes
        z_all = self.fc_enc(x)
        z_app = z_all[:, :self.z_dim_app]
        z_all = z_all[:, self.z_dim_app:]
        z_all = z_all.view(self.batch_size, -1, 3)
        z_gaze_enc = z_all[:, :self.z_dim_gaze, :]
        z_head_enc = z_all[:, self.z_dim_gaze:, :]

        z_gaze_enc = z_gaze_enc.view(self.batch_size, -1, 3)
        z_head_enc = z_head_enc.view(self.batch_size, -1, 3)
        return [z_app, z_gaze_enc, z_head_enc, x, enc_output_shape]

    def decode_to_image(self, codes):
        z_all = torch.cat([code.view(self.batch_size, -1) for code in codes], dim=1)
        x = self.fc_dec(z_all)
        x = x.view(self.batch_size, self.decoder_input_c, *self.bottleneck_shape)
        x = self.decoder(x)
        return x

    def maybe_do_norm(self, code):
        if self.normalize_3d_codes:
            norm_axis = self.normalize_3d_codes_axis
            assert code.dim() == 3
            assert code.shape[-1] == 3
            if norm_axis == 3:
                b, f, _ = code.shape
                code = code.view(b, -1)
                normalized_code = F.normalize(code, dim=-1)
                return normalized_code.view(b, f, -1)
            else:
                return F.normalize(code, dim=norm_axis)
        return code

    def forward(self, data, loss_functions=None):
        is_inference_time = ('image_b' not in data)
        self.batch_size = data['image_a'].shape[0]

        # Encode input from a
        (z_a_a, ze1_g_a, ze1_h_a, ze1_before_z_a, _) = self.encode_to_z(data, 'a')
        if not is_inference_time:
            z_a_b, ze1_g_b, ze1_h_b, _, _ = self.encode_to_z(data, 'b')

        # Make each row a unit vector through L2 normalization to constrain
        # embeddings to the surface of a hypersphere
        if self.normalize_3d_codes:
            assert ze1_g_a.dim() == ze1_h_a.dim() == 3
            assert ze1_g_a.shape[-1] == ze1_h_a.shape[-1] == 3
            ze1_g_a = self.maybe_do_norm(ze1_g_a)
            ze1_h_a = self.maybe_do_norm(ze1_h_a)
            if not is_inference_time:
                ze1_g_b = self.maybe_do_norm(ze1_g_b)
                ze1_h_b = self.maybe_do_norm(ze1_h_b)

        # Gaze estimation output for image a
        if self.backprop_gaze_to_encoder:
            gaze_features = ze1_g_a.clone().view(self.batch_size, -1)
        else:
            # Detach input embeddings from graph!
            gaze_features = ze1_g_a.detach().view(self.batch_size, -1)
        gaze_a_hat = self.gaze2(F.relu_(self.gaze1(gaze_features)))
        gaze_a_hat = F.normalize(gaze_a_hat, dim=-1)

        output_dict = {
            'gaze_a_hat': gaze_a_hat,
            'z_app': z_a_a,
            'z_gaze_enc': ze1_g_a,
            'z_head_enc': ze1_h_a,
            'canon_z_gaze_a': self.rotate_code(data, ze1_g_a, 'gaze', fr='a'),
            'canon_z_head_a': self.rotate_code(data, ze1_h_a, 'head', fr='a'),
        }
        if 'R_gaze_b' not in data:
            return output_dict

        if not is_inference_time:
            output_dict['canon_z_gaze_b'] = self.rotate_code(data, ze1_g_b, 'gaze', fr='b')
            output_dict['canon_z_head_b'] = self.rotate_code(data, ze1_h_b, 'head', fr='b')

        # Rotate codes
        zd1_g_b = self.rotate_code(data, ze1_g_a, 'gaze', fr='a', to='b')
        zd1_h_b = self.rotate_code(data, ze1_h_a, 'head', fr='a', to='b')
        output_dict['z_gaze_dec'] = zd1_g_b
        output_dict['z_head_dec'] = zd1_h_b

        # Reconstruct
        x_b_hat = self.decode_to_image([z_a_a, zd1_g_b, zd1_h_b])
        output_dict['image_b_hat'] = x_b_hat

        # If loss functions specified, apply them
        if loss_functions is not None:
            losses_dict = OrderedDict()
            for key, func in loss_functions.items():
                losses = func(data, output_dict)  # may be dict or single value
                if isinstance(losses, dict):
                    for sub_key, loss in losses.items():
                        losses_dict[key + '_' + sub_key] = loss
                else:
                    losses_dict[key] = losses
            return output_dict, losses_dict

        return output_dict


class DenseNetEncoder(nn.Module):

    def __init__(self, growth_rate=8, num_blocks=4, num_layers_per_block=4,
                 p_dropout=0.0, compression_factor=1.0,
                 activation_fn=nn.ReLU, normalization_fn=nn.BatchNorm2d):
        super(DenseNetEncoder, self).__init__()
        self.c_at_end_of_each_scale = []

        # Initial down-sampling conv layers
        self.initial = DenseNetInitialLayers(growth_rate=growth_rate,
                                             activation_fn=activation_fn,
                                             normalization_fn=normalization_fn)
        c_now = list(self.children())[-1].c_now
        self.c_at_end_of_each_scale += list(self.children())[-1].c_list

        assert (num_layers_per_block % 2) == 0
        for i in range(num_blocks):
            i_ = i + 1
            # Define dense block
            self.add_module('block%d' % i_, DenseNetBlock(
                c_now,
                num_layers=num_layers_per_block,
                growth_rate=growth_rate,
                p_dropout=p_dropout,
                activation_fn=activation_fn,
                normalization_fn=normalization_fn,
            ))
            c_now = list(self.children())[-1].c_now
            self.c_at_end_of_each_scale.append(c_now)

            # Define transition block if not last layer
            if i < (num_blocks - 1):
                self.add_module('trans%d' % i_, DenseNetTransitionDown(
                    c_now, p_dropout=p_dropout,
                    compression_factor=compression_factor,
                    activation_fn=activation_fn,
                    normalization_fn=normalization_fn,
                ))
                c_now = list(self.children())[-1].c_now
            self.c_now = c_now

    def forward(self, x):
        # Apply initial layers and dense blocks
        for name, module in self.named_children():
            if name == 'initial':
                x, prev_scale_x = module(x)
            else:
                x = module(x)
        return x


class DenseNetDecoder(nn.Module):

    def __init__(self, c_in, growth_rate=8, num_blocks=4, num_layers_per_block=4,
                 p_dropout=0.0, compression_factor=1.0,
                 activation_fn=nn.ReLU, normalization_fn=nn.BatchNorm2d,
                 use_skip_connections_from=None):
        super(DenseNetDecoder, self).__init__()

        self.use_skip_connections = (use_skip_connections_from is not None)
        if self.use_skip_connections:
            c_to_concat = use_skip_connections_from.c_at_end_of_each_scale
            c_to_concat = list(reversed(c_to_concat))[1:]
        else:
            c_to_concat = [0] * (num_blocks + 2)

        assert (num_layers_per_block % 2) == 0
        c_now = c_in
        for i in range(num_blocks):
            i_ = i + 1
            # Define dense block
            self.add_module('block%d' % i_, DenseNetBlock(
                c_now,
                num_layers=num_layers_per_block,
                growth_rate=growth_rate,
                p_dropout=p_dropout,
                activation_fn=activation_fn,
                normalization_fn=normalization_fn,
                transposed=True,
            ))
            c_now = list(self.children())[-1].c_now

            # Define transition block if not last layer
            if i < (num_blocks - 1):
                self.add_module('trans%d' % i_, DenseNetTransitionUp(
                    c_now, p_dropout=p_dropout,
                    compression_factor=compression_factor,
                    activation_fn=activation_fn,
                    normalization_fn=normalization_fn,
                ))
                c_now = list(self.children())[-1].c_now
                c_now += c_to_concat[i]

        # Last up-sampling conv layers
        self.last = DenseNetDecoderLastLayers(c_now,
                                              growth_rate=growth_rate,
                                              activation_fn=activation_fn,
                                              normalization_fn=normalization_fn,
                                              skip_connection_growth=c_to_concat[-1])
        self.c_now = 1

    def forward(self, x):
        # Apply initial layers and dense blocks
        for name, module in self.named_children():
            x = module(x)
        return x


class DenseNetDecoderLastLayers(nn.Module):

    def __init__(self, c_in, growth_rate=8, activation_fn=nn.ReLU,
                 normalization_fn=nn.BatchNorm2d,
                 skip_connection_growth=0):
        super(DenseNetDecoderLastLayers, self).__init__()
        # First deconv
        self.conv1 = nn.ConvTranspose2d(c_in, 4 * growth_rate, bias=False,
                                        kernel_size=3, stride=2, padding=1,
                                        output_padding=1)
        nn.init.kaiming_normal_(self.conv1.weight.data)

        # Second deconv
        c_in = 4 * growth_rate + skip_connection_growth
        self.norm2 = normalization_fn(c_in, track_running_stats=False).to(device)
        self.act = activation_fn(inplace=True)
        self.conv2 = nn.ConvTranspose2d(c_in, 2 * growth_rate, bias=False,
                                        kernel_size=3, stride=2, padding=1,
                                        output_padding=1)
        nn.init.kaiming_normal_(self.conv2.weight.data)

        # Final conv
        c_in = 2 * growth_rate
        c_out = 3
        self.norm3 = normalization_fn(c_in, track_running_stats=False).to(device)
        self.conv3 = nn.Conv2d(c_in, c_out, bias=False,
                               kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight.data)
        self.c_now = c_out

    def forward(self, x):
        x = self.conv1(x)
        #
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        #
        x = self.norm3(x)
        x = self.act(x)
        x = self.conv3(x)
        return x


class DenseNetTransitionUp(nn.Module):

    def __init__(self, c_in, compression_factor=0.1, p_dropout=0.1,
                 activation_fn=nn.ReLU, normalization_fn=nn.BatchNorm2d):
        super(DenseNetTransitionUp, self).__init__()
        c_out = int(compression_factor * c_in)
        self.norm = normalization_fn(c_in, track_running_stats=False).to(device)
        self.act = activation_fn(inplace=True)
        self.conv = nn.ConvTranspose2d(c_in, c_out, kernel_size=3,
                                       stride=2, padding=1, output_padding=1,
                                       bias=False).to(device)
        nn.init.kaiming_normal_(self.conv.weight.data)
        self.drop = nn.Dropout2d(p=p_dropout) if p_dropout > 1e-5 else None
        self.c_now = c_out

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x
