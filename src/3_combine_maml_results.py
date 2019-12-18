#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello.
# --------------------------------------------------------

import argparse
import os
import pickle
import re
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter

pickles_to_process = OrderedDict([
    ('GazeCapture (test)', 'predictions_gc.pkl'),
    ('MPIIGaze', 'predictions_mpi.pkl'),
])


def process_dir(input_dir, meta_learner_identifier):
    """Process experiment directory."""
    selected_dirs = OrderedDict()
    candidate_dirs = sorted([
        d for d in os.listdir(input_dir) if os.path.isdir(input_dir + '/' + d)
    ])
    for exp_dir in candidate_dirs:
        maml_dirs = sorted([
            p for p in os.listdir('%s/%s' % (input_dir, exp_dir))
            if re.match(r'^%s_\d{2,4}$' % meta_learner_identifier, p)
        ], key=lambda x: int(x.split('_')[-1]))
        if len(maml_dirs) > 0:
            selected_dirs[exp_dir] = [  # get full paths
                '%s/%s/%s' % (input_dir, exp_dir, p)
                for p in maml_dirs
            ]

    for exp_dir, maml_dirs in selected_dirs.items():
        for dataset, pkl_fname in pickles_to_process.items():
            data = get_all_data(maml_dirs, fname=pkl_fname)

            output_path = '%s/%s %s %s.pdf' % (input_dir, exp_dir,
                                               meta_learner_identifier, dataset)
            plot_mean_error_with_bars(dataset, data, output_path,
                                      meta_learner_identifier)


def get_all_data(all_dirs, fname):
    """Process individual outputs for different k."""
    all_data = OrderedDict()
    for d in all_dirs:
        k = int(d.split('_')[-1])
        ifpath = '%s/%s' % (d, fname)
        if os.path.isfile(ifpath):
            with open(ifpath, 'rb') as f:
                all_data[k] = pickle.load(f)
        else:
            print('Skipping %s' % ifpath)
    return all_data


def common_post(dataset, output_path):
    plt.title(dataset)
    plt.xlabel('k')
    plt.ylabel('Mean Test Error')

    plt.grid()
    plt.tight_layout()

    plt.savefig(output_path)
    print('> Wrote to %s' % output_path)


def plot_mean_error_with_bars(dataset, data, output_path,
                              meta_learner_identifier):
    """Plot standard deviation of mean errors over trials."""
    # Pick out errors from people into single list
    errors = [
        (
            k,
            np.concatenate([
                np.concatenate([
                    trial_data['errors'].reshape(-1, 1)
                    for trial_data in person_data
                ], axis=1)
                for person_data in k_data.values()
            ], axis=0),
        )
        for k, k_data in data.items()
    ]

    ks = [k for k, _ in errors]
    ys = [np.mean(y.reshape(-1)) for _, y in errors]
    es = [np.std(np.mean(y, axis=0)) for _, y in errors]
    print('means:  ', ys)
    print('stddev: ', es)

    plt.clf()
    plt.errorbar(ks, ys, yerr=es, fmt='.-', capsize=5)
    common_post(dataset, output_path)

    # Write means to file
    np.savetxt(output_path[:-3] + 'txt', np.vstack([
        np.array(ks).reshape(1, -1),
        np.array(ys).reshape(1, -1),
        np.array(es).reshape(1, -1),
    ]), fmt='%f')

    # Write means to tensorboard
    tensorboard = SummaryWriter(os.path.dirname(output_path))
    for k, e in zip(ks, ys):
        tensorboard.add_scalar(
            'meta-test-final/%s/%s' % (meta_learner_identifier, dataset), e, k)
    tensorboard.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge MAML outputs on NGC')
    parser.add_argument('input_dir', type=str,
                        help='Training output directory to source MAML predictions from.')
    parser.add_argument('--meta-learner', type=str, choices=['MAML', 'NONE'],
                        default='MAML', help='Select meta learning output to use')
    args = parser.parse_args()
    process_dir(args.input_dir, args.meta_learner)
