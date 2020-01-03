#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello.
# --------------------------------------------------------

import os
import torch
import numpy as np
from torch.utils.data import Dataset

import cv2 as cv
import h5py


class HDFDataset(Dataset):
    """Dataset from HDF5 archives formed of 'groups' of specific persons."""

    def __init__(self, hdf_file_path,
                 prefixes=None,
                 get_2nd_sample=False,
                 pick_exactly_per_person=None,
                 pick_at_least_per_person=None):
        assert os.path.isfile(hdf_file_path)
        self.get_2nd_sample = get_2nd_sample
        self.pick_exactly_per_person = pick_exactly_per_person
        self.hdf_path = hdf_file_path
        self.hdf = None  # h5py.File(hdf_file, 'r')

        with h5py.File(self.hdf_path, 'r', libver='latest', swmr=True) as h5f:
            hdf_keys = sorted(list(h5f.keys()))
            self.prefixes = hdf_keys if prefixes is None else prefixes
            if pick_exactly_per_person is not None:
                assert pick_at_least_per_person is None
                # Pick exactly x many entries from front of group
                self.prefixes = [
                    k for k in self.prefixes if k in h5f
                    and len(next(iter(h5f[k].values()))) >= pick_exactly_per_person
                ]
                self.index_to_query = sum([
                    [(prefix, i) for i in range(pick_exactly_per_person)]
                    for prefix in self.prefixes
                ], [])
            elif pick_at_least_per_person is not None:
                assert pick_exactly_per_person is None
                # Pick people for which there exists at least x many entries
                self.prefixes = [
                    k for k in self.prefixes if k in h5f
                    and len(next(iter(h5f[k].values()))) >= pick_at_least_per_person
                ]
                self.index_to_query = sum([
                    [(prefix, i) for i in range(len(next(iter(h5f[prefix].values()))))]
                    for prefix in self.prefixes
                ], [])
            else:
                # Pick all entries of person
                self.prefixes = [  # to address erroneous inputs
                    k for k in self.prefixes if k in h5f
                    and len(next(iter(h5f[k].values()))) > 0
                ]
                self.index_to_query = sum([
                    [(prefix, i) for i in range(len(next(iter(h5f[prefix].values()))))]
                    for prefix in self.prefixes
                ], [])

    def __len__(self):
        return len(self.index_to_query)

    def close_hdf(self):
        if self.hdf is not None:
            self.hdf.close()
            self.hdf = None

    def preprocess_image(self, image):
        ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv.equalizeHist(ycrcb[:, :, 0])
        image = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB)
        image = np.transpose(image, [2, 0, 1])  # Colour image
        image = 2.0 * image / 255.0 - 1
        return image

    def preprocess_entry(self, entry):
        for key, val in entry.items():
            if isinstance(val, np.ndarray):
                entry[key] = torch.from_numpy(val.astype(np.float32))
            elif isinstance(val, int):
                # NOTE: maybe ints should be signed and 32-bits sometimes
                entry[key] = torch.tensor(val, dtype=torch.int16, requires_grad=False)
        return entry

    def __getitem__(self, idx):
        if self.hdf is None:  # Need to lazy-open this to avoid read error
            self.hdf = h5py.File(self.hdf_path, 'r', libver='latest', swmr=True)

        # Pick entry a and b from same person
        key_a, idx_a = self.index_to_query[idx]
        group_a = self.hdf[key_a]
        group_b = group_a
        all_indices = list(range(len(next(iter(group_a.values())))))
        all_indices_but_a = np.delete(all_indices, idx_a)
        idx_b = np.random.choice(all_indices_but_a)

        def retrieve(group, index):
            eyes = self.preprocess_image(group['pixels'][index, :])
            g = group['labels'][index, :2]
            h = group['labels'][index, 2:4]
            return eyes, g, h

        # Functions to calculate relative rotation matrices for gaze dir. and head pose
        def R_x(theta):
            sin_ = np.sin(theta)
            cos_ = np.cos(theta)
            return np.array([
                [1., 0., 0.],
                [0., cos_, -sin_],
                [0., sin_, cos_]
            ]). astype(np.float32)

        def R_y(phi):
            sin_ = np.sin(phi)
            cos_ = np.cos(phi)
            return np.array([
                [cos_, 0., sin_],
                [0., 1., 0.],
                [-sin_, 0., cos_]
            ]). astype(np.float32)

        def vector_to_pitchyaw(vectors):
            n = vectors.shape[0]
            out = np.empty((n, 2))
            vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
            out[:, 0] = np.arcsin(vectors[:, 1])  # theta
            out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
            return out

        def pitchyaw_to_vector(pitchyaws):
            n = pitchyaws.shape[0]
            sin = np.sin(pitchyaws)
            cos = np.cos(pitchyaws)
            out = np.empty((n, 3))
            out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
            out[:, 1] = sin[:, 0]
            out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
            return out

        def calculate_rotation_matrix(e):
            return np.matmul(R_y(e[1]), R_x(e[0]))

        # Grab 1st (input) entry
        eyes_a, g_a, h_a = retrieve(group_a, idx_a)
        entry = {
            'key': key_a,
            'key_index': self.prefixes.index(key_a),
            'image_a': eyes_a,
            'gaze_a': g_a,
            'head_a': h_a,
            'R_gaze_a': calculate_rotation_matrix(g_a),
            'R_head_a': calculate_rotation_matrix(h_a),
        }

        if self.get_2nd_sample:
            # Grab 2nd entry from same person
            eyes_b, g_b, h_b = retrieve(group_b, idx_b)
            entry['image_b'] = eyes_b
            entry['gaze_b'] = g_b
            entry['head_b'] = h_b
            entry['R_gaze_b'] = calculate_rotation_matrix(entry['gaze_b'])
            entry['R_head_b'] = calculate_rotation_matrix(entry['head_b'])

        return self.preprocess_entry(entry)
