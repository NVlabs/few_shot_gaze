#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello.
# --------------------------------------------------------

from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F


def nn_batch_angular_distance(a, b):
    sim = F.cosine_similarity(a, b, dim=-1, eps=1e-6)
    sim = F.hardtanh(sim, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.mean(torch.acos(sim) * (180 / np.pi), dim=1)


class AllFrontalsEqualLoss(object):

    def __call__(self, input_dict, output_dict):
        # Perform for each gaze and head modes
        loss_terms = OrderedDict()
        for mode in ['gaze', 'head']:
            # Calculate the mean 3D frontalized embedding
            all_embeddings = torch.cat([
                output_dict['canon_z_' + mode + '_a'],
                output_dict['canon_z_' + mode + '_b'],
            ], dim=0)
            mean_embedding = torch.mean(all_embeddings, dim=0)

            # Final calculate and reduce to single loss term
            loss_terms[mode] = torch.std(
                nn_batch_angular_distance(mean_embedding, all_embeddings)
            )
        return loss_terms
