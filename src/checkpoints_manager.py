#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello.
# --------------------------------------------------------


import logging
import os

import torch

ckpt_extension = '.pth.tar'
ckpt_fmtstring = 'at_step_%07d' + ckpt_extension


def step_number_from_fname(fpath):
    fname = fpath.split('/')[-1]
    stem = fname.split('.')[0]
    return int(stem.split('_')[-1])


class CheckpointsManager(object):

    def __init__(self, network, output_dir):
        self.network = network
        self.output_dir = os.path.realpath(output_dir + '/checkpoints')

    @property
    def all_available_checkpoint_files(self):
        if not os.path.isdir(self.output_dir):
            return []
        fpaths = [
            (step_number_from_fname(p), self.output_dir + '/' + p)
            for p in os.listdir(self.output_dir)
            if os.path.isfile(self.output_dir + '/' + p)
            and p.endswith(ckpt_extension)
        ]
        fpaths = sorted(fpaths)  # sort by step number
        return fpaths

    def load_last_checkpoint(self, local_rank):
        available_fpaths = self.all_available_checkpoint_files
        if len(available_fpaths) > 0:
            step_number, fpath = available_fpaths[-1]
            if local_rank == 0:
                logging.info('Found weights file: %s' % fpath)
            loaded_step_number = self.load_checkpoint(step_number, fpath,
                                                      local_rank)
            return loaded_step_number
        return 0

    def load_checkpoint(self, step_number, checkpoint_fpath, local_rank):
        assert os.path.isfile(checkpoint_fpath)
        weights = torch.load(checkpoint_fpath)

        # If was stored using DataParallel but being read on 1 GPU
        if torch.cuda.device_count() == 1:
            if next(iter(weights.keys())).startswith('module.'):
                weights = dict([(k[7:], v) for k, v in weights.items()])

        self.network.load_state_dict(weights, local_rank)
        if local_rank == 0:
            logging.info('Loaded known model weights at step %d' % step_number)
        return step_number

    def save_checkpoint(self, step_number):
        assert os.path.isdir(os.path.abspath(self.output_dir + '/../'))
        fname = ckpt_fmtstring % step_number
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        ofpath = '%s/%s' % (self.output_dir, fname)
        torch.save(self.network.state_dict(), ofpath)
        torch.cuda.empty_cache()
