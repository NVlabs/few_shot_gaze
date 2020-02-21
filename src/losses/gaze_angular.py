#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello.
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F


def nn_angular_distance(a, b):
    sim = F.cosine_similarity(a, b, eps=1e-6)
    sim = F.hardtanh(sim, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(sim) * (180 / np.pi)


class GazeAngularLoss(object):

    def __init__(self, key_true='gaze_a', key_pred='gaze_a_hat'):
        self.key_true = key_true
        self.key_pred = key_pred

    def __call__(self, input_dict, output_dict):
        def pitchyaw_to_vector(pitchyaws):
            sin = torch.sin(pitchyaws)
            cos = torch.cos(pitchyaws)
            return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], 1)
        y = pitchyaw_to_vector(input_dict[self.key_true]).detach()
        y_hat = output_dict[self.key_pred]
        if y_hat.shape[1] == 2:
            y_hat = pitchyaw_to_vector(y_hat)
        return torch.mean(nn_angular_distance(y, y_hat))
