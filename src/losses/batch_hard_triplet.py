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


def nn_batch_euclidean_distance(a, b):
    return torch.mean(torch.norm(a - b, dim=-1, p='fro'), dim=-1)


class BatchHardTripletLoss(object):

    def __init__(self, distance_type, margin=0.0):
        self.margin = margin

        # Select distance function
        self.distance_type = distance_type
        self.distance_fn = None
        if distance_type == 'angular':
            self.distance_fn = nn_batch_angular_distance
        elif distance_type == 'euclidean':
            self.distance_fn = nn_batch_euclidean_distance
        else:
            raise ValueError('Unknown triplet loss distance type: ' + distance_type)

        # Zero loss tensor for when no triplet can be found
        self.zero_loss = torch.tensor(0, dtype=torch.float, requires_grad=False,
                                      device="cuda" if torch.cuda.is_available() else "cpu")

    def construct_person_identicality(self, input_dict):
        """
        Construct a binary matrix which describes whether for a specific matrix
        element, the row-column position mean that the distance is measured
        between (inter) or within (intra) people.
        The value is 1 for the case where person identity is identical,
        and 0 when row and column indices refer to different people.
        """
        num_entries = 2 * len(input_dict['key_index'])
        person_indices = input_dict['key_index'].repeat(2).view(-1, 1)
        person_indices = person_indices.repeat(1, num_entries)
        identicality = (person_indices == person_indices.t()).byte()
        inv_identicality = 1 - identicality
        return identicality.float(), inv_identicality.float()

    def calculate_pairwise_distances(self, output_dict, mode):
        """
        For all given pairs in a batch, calculate distances with selected
        function, pairwise. This means that for a given batch of size B,
        there are 2*B entries. We will calculate (2B)**2 distances.
        """
        num_entries = 2 * len(output_dict['canon_z_' + mode + '_a'])
        all_embeddings = torch.cat([
            output_dict['canon_z_' + mode + '_a'],
            output_dict['canon_z_' + mode + '_b'],
        ], dim=0)
        a = all_embeddings.view(num_entries, 1, -1, 3).repeat(1, num_entries, 1, 1)
        a = a.view(num_entries * num_entries, -1, 3)
        b = all_embeddings.repeat(num_entries, 1, 1)
        return self.distance_fn(a, b).view(num_entries, num_entries)

    def select_hard_triplets(self, dist_grid, person_identicality,
                             inv_person_identicality, selected_row_indices):
        """
        In this function, we select the largest inter-person distance for each
        input entry, and the smallest non-zero intra-person distance for each
        input entry. We only select entries which form valid triplets.
        """
        dist_same = dist_grid * person_identicality
        dist_same_max, _ = torch.max(dist_same, dim=1)

        dist_diff = dist_grid * inv_person_identicality
        dist_diff[dist_diff < 1e-6] = 1e6  # set some large value to zero values
        dist_diff_min, _ = torch.min(dist_diff, dim=1)

        if len(selected_row_indices) < len(dist_grid):
            dist_same_max = torch.take(dist_same_max, selected_row_indices)
            dist_diff_min = torch.take(dist_diff_min, selected_row_indices)
        return dist_same_max, dist_diff_min

    def __call__(self, input_dict, output_dict):
        # Calculate masks
        person_identicality, inv_person_identicality = \
            self.construct_person_identicality(input_dict)

        # Select only those that have both same and diff entries
        # Basically ensure that there are valid triplets
        num_per_row_same = torch.sum(person_identicality.byte(), dim=-1) - 1
        num_per_row_diff = torch.sum(inv_person_identicality.byte(), dim=-1)
        num_per_row_both = num_per_row_same * num_per_row_diff
        selected_row_indices = torch.nonzero(num_per_row_both)

        # Perform for each gaze and head modes
        loss_terms = OrderedDict()
        for mode in ['gaze', 'head']:
            # Calculate pairwise distances
            pairwise_distances = self.calculate_pairwise_distances(output_dict, mode)

            # Reduce to hard samples
            d_positive, d_negative = self.select_hard_triplets(
                pairwise_distances, person_identicality,
                inv_person_identicality, selected_row_indices,
            )

            # Final calculate and reduce to single loss term
            stem = mode + '_' + self.distance_type
            if len(d_positive) > 0:
                loss_terms[stem] = torch.mean(F.softplus(d_positive - d_negative + self.margin))
                loss_terms[stem + '_d_within'] = torch.mean(d_positive)
                loss_terms[stem + '_d_between'] = torch.mean(d_negative)
            else:
                loss_terms[stem] = self.zero_loss
                loss_terms[stem + '_d_within'] = self.zero_loss
                loss_terms[stem + '_d_between'] = self.zero_loss
        return loss_terms
