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
    # The inputs here are of shape: B x F x 3
    # where B: batch size
    #       F: no. of features
    # we would like to compare each corresponding feature separately
    assert a.dim() == b.dim() == 3
    assert a.shape[-1] == b.shape[-1] == 3
    sim = F.cosine_similarity(a, b, dim=-1, eps=1e-6)
    # We now have distances with shape B x F

    # Ensure no NaNs occur due to the input to the arccos function
    sim = F.hardtanh(sim, -1.0 + 1e-6, 1.0 - 1e-6)

    # Now, we want to convert the similarity measure to degrees and calculate
    # a single scalar distance value per entry in the batch
    batch_distance = torch.mean(torch.acos(sim) * (180 / np.pi), dim=1)

    # The output is of length B
    assert batch_distance.dim() == 1
    return batch_distance


def nn_batch_euclidean_distance(a, b):
    # The inputs here are of shape: B x F x 3
    # Let's compare each 3D unit vector feature separately
    assert a.dim() == b.dim() == 3
    assert a.shape[-1] == b.shape[-1] == 3
    featurewise_dists = torch.norm(a - b, dim=-1, p='fro')

    # Calculate a single scalar distance value per entry in the batch
    entrywise_dists = torch.mean(featurewise_dists, dim=-1)
    return entrywise_dists


class EmbeddingConsistencyLoss(object):

    def __init__(self, distance_type):
        # Select distance function
        self.distance_type = distance_type
        self.distance_fn = None
        if distance_type == 'angular':
            self.distance_fn = nn_batch_angular_distance
        elif distance_type == 'euclidean':
            self.distance_fn = nn_batch_euclidean_distance
        else:
            raise ValueError('Unknown triplet loss distance type: ' + distance_type)

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
        return identicality.float()

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

    def select_max_distances(self, dist_grid, person_identicality):
        """
        In this function, we select the largest inter-person distance for each
        input entry, and the smallest non-zero intra-person distance for each
        input entry. We only select entries which form valid triplets.
        """
        dist_same = dist_grid * person_identicality
        dist_same_max, _ = torch.max(dist_same, dim=1)
        return dist_same_max

    def __call__(self, input_dict, output_dict):
        # Calculate masks
        person_identicality = self.construct_person_identicality(input_dict)

        # Perform for each gaze and head modes
        loss_terms = OrderedDict()
        for mode in ['gaze', 'head']:
            # Calculate pairwise distances
            pairwise_distances = self.calculate_pairwise_distances(output_dict, mode)

            # Reduce to hard samples
            d_positive = self.select_max_distances(pairwise_distances, person_identicality)

            # Final calculate and reduce to single loss term
            stem = mode + '_' + self.distance_type
            loss_terms[stem] = torch.mean(d_positive)
        return loss_terms
