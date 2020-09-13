#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello.
# --------------------------------------------------------


import argparse
parser = argparse.ArgumentParser(description='Train DT-ED')

# architecture specification
parser.add_argument('--densenet-growthrate', type=int, default=32,
                    help='growth rate of encoder/decoder base densenet archi. (default: 32)')
parser.add_argument('--z-dim-app', type=int, default=64,
                    help='size of 1D latent code for appearance (default: 64)')
parser.add_argument('--z-dim-gaze', type=int, default=2,
                    help='size of 2nd dim. of 3D latent code for each gaze direction (default: 2)')
parser.add_argument('--z-dim-head', type=int, default=16,
                    help='size of 2nd dim. of 3D latent code for each head rotation (default: 16)')
parser.add_argument('--decoder-input-c', type=int, default=32,
                    help='size of feature map stack as input to decoder (default: 32)')

parser.add_argument('--normalize-3d-codes', action='store_true',
                    help='normalize rows of 3D latent codes')
parser.add_argument('--normalize-3d-codes-axis', default=1, type=int, choices=[1, 2, 3],
                    help='axis over which to normalize 3D latent codes')

parser.add_argument('--triplet-loss-type', choices=['angular', 'euclidean'],
                    help='Apply triplet loss with selected distance metric')
parser.add_argument('--triplet-loss-margin', type=float, default=0.0,
                    help='Triplet loss margin')
parser.add_argument('--triplet-regularize-d-within', action='store_true',
                    help='Regularize triplet loss by mean within-person distance')

parser.add_argument('--all-equal-embeddings', action='store_true',
                    help='Apply loss to make all frontalized embeddings similar')

parser.add_argument('--embedding-consistency-loss-type',
                    choices=['angular', 'euclidean'], default=None,
                    help='Apply embedding_consistency loss with selected distance metric')
parser.add_argument('--embedding-consistency-loss-warmup-samples',
                    type=int, default=1000000,
                    help='Start from 0.0 and warm up embedding consistency loss until n samples')

parser.add_argument('--backprop-gaze-to-encoder', action='store_true',
                    help='Add gaze loss term to single loss and backprop to entire network.')

parser.add_argument('--coeff-l1-recon-loss', type=float, default=1.0,
                    help='Weight/coefficient for L1 reconstruction loss term')
parser.add_argument('--coeff-gaze-loss', type=float, default=0.1,
                    help='Weight/coefficient for gaze direction loss term')
parser.add_argument('--coeff-embedding_consistency-loss', type=float, default=2.0,
                    help='Weight/coefficient for embedding_consistency loss term')

# training
parser.add_argument('--pick-exactly-per-person', type=int, default=None,
                    help='Pick exactly this many entries per person for training.')
parser.add_argument('--pick-at-least-per-person', type=int, default=400,
                    help='Only pick person for training if at least this many entries.')
parser.add_argument('--use-apex', action='store_true',
                    help='Use half-precision floating points via the apex library.')
parser.add_argument('--base-lr', type=float, default=0.00005, metavar='LR',
                    help='learning rate (to be multiplied with batch size) (default: 0.00005)')
parser.add_argument('--warmup-period-for-lr', type=int, default=1000000, metavar='LR',
                    help=('no. of data entries (not batches) to have processed '
                          + 'when stopping  gradual ramp up of LR (default: 1000000)'))
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='training batch size (default: 128)')
parser.add_argument('--decay-interval', type=int, default=0, metavar='N',
                    help='iterations after which to decay the learning rate (default: 0)')
parser.add_argument('--decay', type=float, default=0.8, metavar='decay',
                    help='learning rate decay multiplier (default: 0.8)')
parser.add_argument('--num-training-epochs', type=float, default=20, metavar='N',
                    help='number of steps to train (default: 20)')
parser.add_argument('--l2-reg', type=float, default=1e-4,
                    help='l2 weights regularization coefficient (default: 1e-4)')
parser.add_argument('--print-freq-train', type=int, default=20, metavar='N',
                    help='print training statistics after every N iterations (default: 20)')
parser.add_argument('--print-freq-test', type=int, default=5000, metavar='N',
                    help='print test statistics after every N iterations (default: 5000)')
parser.add_argument('--distributed', dest = "distributed", action = 'store_true',
                    help = 'Use distributed computing in training.')
parser.add_argument('--local_rank', default = 0, type = int)

# data
parser.add_argument('--mpiigaze-file', type=str, default='../preprocess/outputs/MPIIGaze.h5',
                    help='Path to MPIIGaze dataset in HDF format.')
parser.add_argument('--gazecapture-file', type=str, default='../preprocess/outputs/GazeCapture.h5',
                    help='Path to GazeCapture dataset in HDF format.')
parser.add_argument('--test-subsample', type=float, default=1.0,
                    help='proportion of test set to use (default: 1.0)')
parser.add_argument('--num-data-loaders', type=int, default=0, metavar='N',
                    help='number of data loading workers (default: 0)')

# logging
parser.add_argument('--use-tensorboard', action='store_true', default=False,
                    help='create tensorboard logs (stored in the args.save_path directory)')
parser.add_argument('--save-path', type=str, default='.',
                    help='path to save network parameters (default: .)')
parser.add_argument('--show-warnings', action='store_true', default=False,
                    help='show default Python warnings')

# image saving
parser.add_argument('--save-freq-images', type=int, default=1000,
                    help='save sample images after every N iterations (default: 1000)')
parser.add_argument('--save-image-samples', type=int, default=100,
                    help='Save image outputs for N samples per dataset (default: 100)')

# evaluation / prediction of outputs
parser.add_argument('--skip-training', action='store_true',
                    help='skip training to go straight to prediction generation')
parser.add_argument('--generate-predictions', action='store_true',
                    help='skip training to go straight to prediction generation')
parser.add_argument('--eval-batch-size', type=int, default=512, metavar='N',
                    help='evaluation batch size (default: 512)')

args = parser.parse_args()

import h5py
import numpy as np
from collections import OrderedDict
import gc
import json
import time
import os

import moviepy.editor as mpy

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
if args.distributed:
    from torch.utils.data.distributed import DistributedSampler
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend = 'nccl', init_method = 'env://')
    world_size = torch.distributed.get_world_size()
else:
    world_size = torch.cuda.device_count()

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

from data import HDFDataset

# Set device
if args.distributed:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Ignore warnings
if not args.show_warnings:
    import warnings
    warnings.filterwarnings('ignore')

#############################
# Sanity check some arguments

if args.embedding_consistency_loss_type is not None:
    assert args.triplet_loss_type is None
if args.triplet_loss_type is not None:
    assert args.embedding_consistency_loss_type is None

if (args.triplet_loss_type == 'angular'
        or args.embedding_consistency_loss_type == 'angular'):
    assert args.normalize_3d_codes is True
elif (args.triplet_loss_type == 'euclidean'
        or args.embedding_consistency_loss_type == 'euclidean'):
    assert args.normalize_3d_codes is False


def embedding_consistency_loss_weight_at_step(current_step):
    final_value = args.coeff_embedding_consistency_loss
    if args.embedding_consistency_loss_warmup_samples is None:
        return final_value
    warmup_steps = int(args.embedding_consistency_loss_warmup_samples / batch_size_global)
    if current_step <= warmup_steps:
        return (final_value / warmup_steps) * current_step
    else:
        return final_value


#####################################################
# Calculate how to handle learning rate at given step

global batch_size_global
batch_size_global = args.batch_size * world_size

max_lr = args.base_lr * batch_size_global
ramp_up_until_step = int(args.warmup_period_for_lr / batch_size_global)
ramp_up_a = (max_lr - args.base_lr) / ramp_up_until_step
ramp_up_b = args.base_lr


def learning_rate_at_step(current_step):
    if current_step <= ramp_up_until_step:
        return ramp_up_a * current_step + ramp_up_b
    elif args.decay_interval != 0:
        return np.power(args.decay, int((current_step - ramp_up_until_step)
                                        / args.decay_interval))
    else:
        return max_lr


def update_learning_rate(current_step):
    global optimizer
    lr = learning_rate_at_step(current_step)
    all_param_groups = optimizer.param_groups
    for i, param_group in enumerate(all_param_groups):
        if i == 0:  # Don't do it for the gaze-related weights
            param_group['lr'] = lr


################################################
# Create network
from models import DTED
network = DTED(
    growth_rate=args.densenet_growthrate,
    z_dim_app=args.z_dim_app,
    z_dim_gaze=args.z_dim_gaze,
    z_dim_head=args.z_dim_head,
    decoder_input_c=args.decoder_input_c,
    normalize_3d_codes=args.normalize_3d_codes,
    normalize_3d_codes_axis=args.normalize_3d_codes_axis,
    use_triplet=args.triplet_loss_type is not None,
    backprop_gaze_to_encoder=args.backprop_gaze_to_encoder,
)
if args.distributed:
        if args.local_rank == 0:
            logging.info(network)
else:
    logging.info(network)

################################################
# Transfer on the GPU before constructing and optimizer
if args.distributed:
    network = network.cuda()
else:
    network = network.to(device)

################################################
# Build optimizers
if args.use_apex:
    from apex.optimizers import FusedSGD
    SGD = FusedSGD
else:
    SGD = optim.SGD

gaze_lr = 1.0 * args.base_lr
if args.backprop_gaze_to_encoder:
    optimizer = SGD(
        [
            {'params': [p for n, p in network.named_parameters() if not n.startswith('gaze')]},
            {
                'params': [p for n, p in network.named_parameters() if n.startswith('gaze')],
                'lr': gaze_lr,
            },
        ],
        lr=args.base_lr, momentum=0.9,
        nesterov=True, weight_decay=args.l2_reg)
else:
    optimizer = SGD(
        [p for n, p in network.named_parameters() if not n.startswith('gaze')],
        lr=args.base_lr, momentum=0.9,
        nesterov=True, weight_decay=args.l2_reg,
    )

    # one additional optimizer for just gaze estimation head
    gaze_optimizer = SGD(
        [p for n, p in network.named_parameters() if n.startswith('gaze')],
        lr=gaze_lr, momentum=0.9,
        nesterov=True, weight_decay=args.l2_reg,
    )

# Wrap optimizer instances with AMP
if args.use_apex:
    from apex import amp
    optimizers = ([optimizer]
                  if args.backprop_gaze_to_encoder
                  else [optimizer, gaze_optimizer])
    network, optimizers = amp.initialize(network, optimizers,
                                         opt_level='O1', num_losses=len(optimizers))
    if args.backprop_gaze_to_encoder:
        optimizer = optimizers[0]
    else:
        optimizer, gaze_optimizer = optimizers

if not args.distributed or args.local_rank == 0:
    logging.info('Initialized optimizer(s)')

################################################
# Implement data parallel training on multiple GPUs if available
if args.distributed:
    network = DDP(network, device_ids=[args.local_rank])
    if args.local_rank == 0:
        logging.info('Using %d GPUs! with DDP' % world_size)
        seed = np.random.randint(1e4)
        seed = (seed + torch.distributed.get_rank()) % 2**32
else:
    if torch.cuda.device_count() > 1:
        network = nn.DataParallel(network)
    logging.info('Using %d GPUs! with DP' % world_size)

################################################
# Define loss functions
from losses import (ReconstructionL1Loss, GazeAngularLoss, BatchHardTripletLoss,
                    AllFrontalsEqualLoss, EmbeddingConsistencyLoss)

loss_functions = OrderedDict()
loss_functions['recon_l1'] = ReconstructionL1Loss(suffix='b')
loss_functions['gaze'] = GazeAngularLoss()

if args.triplet_loss_type is not None:
    loss_functions['triplet'] = BatchHardTripletLoss(
        distance_type=args.triplet_loss_type,
        margin=args.triplet_loss_margin,
    )

if args.all_equal_embeddings:
    loss_functions['all_equal'] = AllFrontalsEqualLoss()

if args.embedding_consistency_loss_type is not None:
    loss_functions['embedding_consistency'] = EmbeddingConsistencyLoss(
        distance_type=args.embedding_consistency_loss_type,
    )

################################################
# Create the train and test datasets.
# We train on the GazeCapture training set
# and test on the val+test set, and the entire MPIIGaze.
all_data = OrderedDict()


def worker_init_fn(worker_id):
    # Custom worker init to not repeat pairs
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# Load GazeCapture prefixes with train/val/test split spec.
with open('./gazecapture_split.json', 'r') as f:
    all_gc_prefixes = json.load(f)

# Define single training dataset
train_tag = 'gc/train'
train_prefixes = all_gc_prefixes['train']
train_dataset = HDFDataset(hdf_file_path=args.gazecapture_file,
                           prefixes=train_prefixes,
                           get_2nd_sample=True,
                           pick_exactly_per_person=args.pick_exactly_per_person,
                           pick_at_least_per_person=args.pick_at_least_per_person,
                           )
global train_dataloader
if args.distributed:
    train_sampler = DistributedSampler(train_dataset,
                                       num_replicas=world_size,
                                       rank=args.local_rank,
                                       shuffle=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  drop_last=True,
                                  num_workers=args.num_data_loaders,
                                  pin_memory=True,
                                  sampler=train_sampler)
else:
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=args.num_data_loaders,
                                  pin_memory=True,
                                  # worker_init_fn=worker_init_fn,
                                  )
all_data[train_tag] = {'dataset': train_dataset, 'dataloader': train_dataloader}

# Define multiple validation/test datasets
for tag, hdf_file, prefixes in [('gc/val', args.gazecapture_file, all_gc_prefixes['val']),
                                ('gc/test', args.gazecapture_file, all_gc_prefixes['test']),
                                ('mpi', args.mpiigaze_file, None),
                                ]:
    # Define dataset structure based on selected prefixes
    dataset = HDFDataset(hdf_file_path=hdf_file,
                         prefixes=prefixes,
                         get_2nd_sample=True)
    subsample = args.test_subsample
    if tag == 'gc/test':  # reduce no. of test samples for this case
        subsample /= 10.0
    if subsample < (1.0 - 1e-6):  # subsample if requested
        dataset = Subset(dataset, np.linspace(
            start=0, stop=len(dataset),
            num=int(subsample * len(dataset)),
            endpoint=False,
            dtype=np.uint32,
        ))
    if args.distributed:
        sampler = DistributedSampler(dataset,
                                     num_replicas=world_size,
                                     rank=args.local_rank,
                                     shuffle=False)
        all_data[tag] = {
            'dataset': dataset,
            'dataloader': DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=2,  # args.num_data_loaders,
                                     pin_memory=True,
                                     sampler=sampler)
        }
    else:
        all_data[tag] = {
            'dataset': dataset,
            'dataloader': DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=2,  # args.num_data_loaders,
                                     pin_memory=True,
                                     worker_init_fn=worker_init_fn),
        }

# Print some stats.
if not args.distributed or args.local_rank == 0:
    for tag, val in all_data.items():
        tag = '[%s]' % tag
        dataset = val['dataset']
        original_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
        num_original_entries = len(original_dataset)
        num_people = len(original_dataset.prefixes)
        logging.info('%10s full set size:           %7d' % (tag, num_original_entries))
        logging.info('%10s current set size:        %7d' % (tag, len(dataset)))
        logging.info('%10s num people:              %7d' % (tag, num_people))
        logging.info('%10s mean entries per person: %7d' % (tag, num_original_entries / num_people))
        logging.info('')

    logging.info('Prepared Datasets')


######################################################
# Utility methods for accessing datasets


def send_data_dict_to_gpu(data):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().to(device, non_blocking=True)
    return data


######################################################
# Pre-collect entries for which to generate images for

for tag, data_dict in all_data.items():
    dataset = data_dict['dataset']
    indices = np.linspace(start=0, stop=len(dataset), endpoint=False,
                          num=args.save_image_samples, dtype=np.uint32)
    retrieved_samples = [dataset[index] for index in indices]
    stacked_samples = {}
    for k in ['image_a', 'face_a', 'R_gaze_a', 'R_head_a']:
        if k in retrieved_samples[0]:
            stacked_samples[k] = torch.stack([s[k] for s in retrieved_samples])
    data_dict['to_visualize'] = stacked_samples

    # Have dataloader re-open HDF to avoid multi-processing related errors.
    original_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    original_dataset.close_hdf()


#################################
# Latent Space Walk Preparations


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


walking_spec = []
for rotation_fn, short, min_d, max_d, num_d in [(R_x, 'x', 45, -45, 15),
                                                (R_y, 'y', -45, 45, 15)]:
    degrees = np.linspace(min_d, max_d, num_d, dtype=np.float32)
    walking_spec.append({
        'name': '%s_%d_%d' % (short, min_d, max_d),
        'matrices': [
            torch.from_numpy(np.repeat(
                np.expand_dims(rotation_fn(np.radians(deg)), 0),
                args.save_image_samples,
                axis=0,
            ))
            for deg in degrees
        ],
    })

identity_rotation = torch.from_numpy(np.repeat(
    np.expand_dims(np.eye(3, dtype=np.float32), 0),
    args.save_image_samples,
    axis=0,
))


def recover_images(x):
    # Every specified iterations save sample images
    # Note: We're doing this separate to Tensorboard to control which input
    #       samples we visualize, and also because Tensorboard is an inefficient
    #       way to store such images.
    x = x.cpu().numpy()
    x = (x + 1.0) * (255.0 / 2.0)
    x = np.clip(x, 0, 255)  # Avoid artifacts due to slight under/overflow
    x = np.transpose(x, [0, 2, 3, 1])  # CHW to HWC
    x = x.astype(np.uint8)
    x = x[:, :, :, ::-1]  # RGB to BGR for OpenCV
    return x


############################
# Load weights if available

from checkpoints_manager import CheckpointsManager
saver = CheckpointsManager(network, args.save_path)
initial_step = saver.load_last_checkpoint(args.local_rank)

######################
# Training step update


class RunningStatistics(object):
    def __init__(self):
        self.losses = OrderedDict()

    def add(self, key, value):
        if key not in self.losses:
            self.losses[key] = []
        self.losses[key].append(value)

    def means(self):
        return OrderedDict([
            (k, np.mean(v)) for k, v in self.losses.items() if len(v) > 0
        ])

    def reset(self):
        for key in self.losses.keys():
            self.losses[key] = []


time_epoch_start = None
num_elapsed_epochs = 0

def reduce_loss(loss):
    loss_clone = loss.clone()
    torch.distributed.all_reduce(loss_clone, op = torch.distributed.ReduceOp.SUM)
    avg_loss = loss_clone / world_size
    return avg_loss

def execute_training_step(current_step):
    global train_data_iterator, time_epoch_start, num_elapsed_epochs
    torch.cuda.synchronize()
    time_iteration_start = time.time()

    # Get data
    try:
        if time_epoch_start is None:
            time_epoch_start = time.time()
        time_batch_fetch_start = time.time()
        input_dict = next(train_data_iterator)
    except StopIteration:
        # Epoch counter and timer
        num_elapsed_epochs += 1
        time_epoch_end = time.time()
        time_epoch_diff = time_epoch_end - time_epoch_start
        if not args.distributed or args.local_rank == 0:
            if args.use_tensorboard:
                    tensorboard.add_scalar('timing/epoch', time_epoch_diff, num_elapsed_epochs)

        # Done with an epoch now...!
        if num_elapsed_epochs % 5 == 0:
            saver.save_checkpoint(current_step)

        np.random.seed()  # Ensure randomness

        # Some cleanup
        train_data_iterator = None
        torch.cuda.empty_cache()
        gc.collect()

        # Restart!
        time_epoch_start = time.time()
        global train_dataloader
        train_data_iterator = iter(train_dataloader)
        time_batch_fetch_start = time.time()
        input_dict = next(train_data_iterator)

    # get the inputs
    input_dict = send_data_dict_to_gpu(input_dict)
    if not args.distributed or args.local_rank == 0:
        running_timings.add('batch_fetch', time.time() - time_batch_fetch_start)

    # zero the parameter gradient
    network.train()
    optimizer.zero_grad()
    if not args.backprop_gaze_to_encoder:
        gaze_optimizer.zero_grad()

    # forward + backward + optimize
    time_forward_start = time.time()
    output_dict, loss_dict = network(input_dict, loss_functions=loss_functions)
    # torch.cuda.synchronize()

    # If doing multi-GPU training, just take an average
    for key, value in loss_dict.items():
        if value.dim() > 0:
            value = torch.mean(value)
            loss_dict[key] = value

    # Construct main loss
    loss_to_optimize = args.coeff_l1_recon_loss * loss_dict['recon_l1']
    if args.triplet_loss_type is not None:
        triplet_losses = []
        triplet_losses = [
            loss_dict['triplet_gaze_' + args.triplet_loss_type],
            loss_dict['triplet_head_' + args.triplet_loss_type],
        ]
        if args.triplet_regularize_d_within:
            triplet_losses += [
                loss_dict['triplet_gaze_%s_d_within' % args.triplet_loss_type],
                loss_dict['triplet_head_%s_d_within' % args.triplet_loss_type],
            ]
        loss_to_optimize += 1.0 * sum(triplet_losses)

    if args.embedding_consistency_loss_type is not None:
        embedding_consistency_losses = [
            loss_dict['embedding_consistency_gaze_' + args.embedding_consistency_loss_type],
            # loss_dict['embedding_consistency_head_' + args.embedding_consistency_loss_type],
        ]
        coeff_embedding_consistency_loss = embedding_consistency_loss_weight_at_step(current_step)
        loss_to_optimize += coeff_embedding_consistency_loss * sum(embedding_consistency_losses)

    if args.all_equal_embeddings:
        loss_to_optimize += sum([
            loss_dict['all_equal_gaze'],
            loss_dict['all_equal_head'],
        ])

    if args.backprop_gaze_to_encoder:
        loss_to_optimize += args.coeff_gaze_loss * loss_dict['gaze']

    # Learning rate ramp-up until specified no. of samples passed, or decay
    update_learning_rate(current_step)

    # Optimize main objective
    if args.use_apex:
        with amp.scale_loss(loss_to_optimize, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss_to_optimize.backward()
    optimizer.step()

    # optimize small gaze part too, separately (if required)
    if not args.backprop_gaze_to_encoder:
        if args.use_apex:
            with amp.scale_loss(loss_dict['gaze'], gaze_optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_dict['gaze'].backward()
        gaze_optimizer.step()

    # Register timing
    time_backward_end = time.time()
    if not args.distributed or args.local_rank == 0:
        running_timings.add('forward_and_backward', time_backward_end - time_forward_start)

    # Store values for logging later
    if args.distributed:
        # Reuce loss across GPUs (strong data to cpu includes cuda synchronization)
        for key, value in loss_dict.items():
            loss_dict[key] = reduce_loss(value).detach().cpu()
    else:
        for key, value in loss_dict.items():
            loss_dict[key] = value.detach().cpu()

    if not args.distributed or args.local_rank == 0:
        for key, value in loss_dict.items():
            running_losses.add(key, value.numpy())
        running_timings.add('iteration', time.time() - time_iteration_start)


####################################
# Test for particular validation set


def execute_test(tag, data_dict):
    test_losses = RunningStatistics()
    with torch.no_grad():
        for input_dict in data_dict['dataloader']:
            network.eval()
            input_dict = send_data_dict_to_gpu(input_dict)
            output_dict, loss_dict = network(input_dict, loss_functions=loss_functions)
            for key, value in loss_dict.items():
                test_losses.add(key, value.detach().cpu().numpy())
    test_loss_means = test_losses.means()
    if not args.distributed or args.local_rank == 0:
        logging.info('Test Losses at [%7d] for %10s: %s' %
                     (current_step + 1, '[' + tag + ']',
                      ', '.join(['%s: %.6f' % v for v in test_loss_means.items()])))
        if args.use_tensorboard:
            for k, v in test_loss_means.items():
                tensorboard.add_scalar('test/%s/%s' % (tag, k), v, current_step + 1)


############
# Main loop

num_training_steps = int(args.num_training_epochs * len(train_dataset) / batch_size_global)
if args.skip_training:
    num_training_steps = 0
else:
    if not args.distributed or args.local_rank == 0:
        logging.info('Training')
        if args.use_tensorboard:
            from tensorboardX import SummaryWriter
            tensorboard = SummaryWriter(log_dir=args.save_path)
    last_training_step = num_training_steps - 1


train_data_iterator = iter(train_dataloader)
if not args.distributed or args.local_rank == 0:
    running_losses = RunningStatistics()
    running_timings = RunningStatistics()
for current_step in range(initial_step, num_training_steps):

    ################
    # Training loop
    execute_training_step(current_step)

    if current_step % args.print_freq_train == args.print_freq_train - 1:
        conv1_wt_lr = optimizer.param_groups[0]['lr']
        if not args.distributed or args.local_rank == 0:
            running_loss_means = running_losses.means()
            logging.info('Losses at [%7d]: %s' %
                             (current_step + 1,
                              ', '.join(['%s: %.5f' % v
                                         for v in running_loss_means.items()])))
            if args.use_tensorboard:
                tensorboard.add_scalar('train_lr', conv1_wt_lr, current_step + 1)
                for k, v in running_loss_means.items():
                    tensorboard.add_scalar('train/' + k, v, current_step + 1)
            running_losses.reset()

    # Print some timing statistics
    if current_step % 100 == 99:
        if not args.distributed or args.local_rank == 0:
            if args.use_tensorboard:
                    for k, v in running_timings.means().items():
                        tensorboard.add_scalar('timing/' + k, v, current_step + 1)
            running_timings.reset()

    # print some memory statistics
    if current_step % 5000 == 0:
        if args.distributed:
            bytes = (torch.cuda.memory_allocated(device=args.local_rank)
                 + torch.cuda.memory_cached(device=args.local_rank))
            logging.info('GPU %d: probably allocated approximately %.2f GB' %
                (args.local_rank, bytes / 1e9))
        else:
            for i in range(torch.cuda.device_count()):
                bytes = (torch.cuda.memory_allocated(device=i)
                         + torch.cuda.memory_cached(device=i))
                logging.info('GPU %d: probably allocated approximately %.2f GB' % (i, bytes / 1e9))

    ###############
    # Testing loop: every specified iterations compute the test statistics
    if (current_step % args.print_freq_test == (args.print_freq_test - 1)
            or current_step == last_training_step):
        network.eval()
        optimizer.zero_grad()
        if not args.backprop_gaze_to_encoder:
            gaze_optimizer.zero_grad()
        torch.cuda.empty_cache()

        for tag, data_dict in list(all_data.items())[1:]:
            execute_test(tag, data_dict)

            # This might help with memory leaks
            torch.cuda.empty_cache()

    #####################
    # Visualization loop

    # Latent space walks (only store latest results)
    if not args.distributed or args.local_rank == 0:
        if (args.save_image_samples > 0
            and (current_step % args.save_freq_images
                 == (args.save_freq_images - 1)
                 or current_step == last_training_step)):
            network.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                for tag, data_dict in all_data.items():

                    def save_images(images, dname, stem):
                        dpath = '%s/walks/%s/%s' % (args.save_path, tag, dname)
                        if not os.path.isdir(dpath):
                            os.makedirs(dpath)
                        for i in range(args.save_image_samples):
                            # Write single image
                            frames = [images[j][i] for j in range(len(images))]
                            # Write video
                            frames = [f[:, :, ::-1] for f in frames]  # BGR to RGB
                            frames += frames[1:-1][::-1]  # continue in reverse
                            clip = mpy.ImageSequenceClip(frames, fps=15)
                            clip.write_videofile('%s/%04d_%s.mp4' % (dpath, i, stem),
                                                 audio=False, threads=8,
                                                 logger=None, verbose=False)

                    for spec in walking_spec:  # Gaze-direction-walk
                        output_images = []
                        for rotation_mat in spec['matrices']:
                            adjusted_input = data_dict['to_visualize'].copy()
                            adjusted_input['R_gaze_b'] = rotation_mat
                            adjusted_input['R_head_b'] = identity_rotation
                            adjusted_input = dict([(k, v.to(device))
                                                   for k, v in adjusted_input.items()])
                            output_dict = network(adjusted_input)
                            output_images.append(recover_images(output_dict['image_b_hat']))
                        save_images(output_images, 'gaze', spec['name'])

                    for spec in walking_spec:  # Head-pose-walk
                        output_images = []
                        for rotation_mat in spec['matrices']:
                            adjusted_input = data_dict['to_visualize'].copy()
                            adjusted_input['R_gaze_b'] = identity_rotation
                            adjusted_input['R_head_b'] = rotation_mat
                            adjusted_input = dict([(k, v.to(device))
                                                   for k, v in adjusted_input.items()])
                            output_dict = network(adjusted_input)
                            output_images.append(recover_images(output_dict['image_b_hat']))
                        save_images(output_images, 'head', spec['name'])

            torch.cuda.empty_cache()

if not args.skip_training:
    if not args.distributed or args.local_rank == 0:
        logging.info('Finished Training')

        # Save model parameters
        saver.save_checkpoint(current_step)
        if args.use_tensorboard:
            tensorboard.close()
            del tensorboard

# Clean up a bit
optimizer.zero_grad()
del (train_dataloader, train_dataset, all_data,
     walking_spec, optimizer, identity_rotation)

#########################################
# Generating predictions with final model

if args.generate_predictions:
    # make sure that DDP is off for generating predictions as multiple
    # processes cannot write together to the same .h5 file.
    assert args.distributed is False

    logging.info('Now generating predictions with final model...')
    all_data = OrderedDict()
    for tag, hdf_file, prefixes in [('gc/train', args.gazecapture_file, all_gc_prefixes['train']),
                                    ('gc/val', args.gazecapture_file, all_gc_prefixes['val']),
                                    ('gc/test', args.gazecapture_file, all_gc_prefixes['test']),
                                    ('mpi', args.mpiigaze_file, None),
                                    ]:
        # Define dataset structure based on selected prefixes
        dataset = HDFDataset(hdf_file_path=hdf_file,
                             prefixes=prefixes,
                             get_2nd_sample=False)
        all_data[tag] = {
            'dataset': dataset,
            'dataloader': DataLoader(dataset,
                                     batch_size=args.eval_batch_size,
                                     shuffle=False,
                                     num_workers=args.num_data_loaders,
                                     pin_memory=True,
                                     worker_init_fn=worker_init_fn),
        }
    logging.info('')
    for tag, val in all_data.items():
        tag = '[%s]' % tag
        dataset = val['dataset']
        num_entries = len(dataset)
        num_people = len(dataset.prefixes)
        logging.info('%10s set size:                %7d' % (tag, num_entries))
        logging.info('%10s num people:              %7d' % (tag, num_people))
        logging.info('%10s mean entries per person: %7d' % (tag, num_entries / num_people))
        logging.info('')

    # every specified iterations compute the test statistics:
    for tag, data_dict in all_data.items():
        current_person_id = None
        current_person_data = {}
        ofpath = '%s/%s_predictions.h5' % (args.save_path, tag.replace('/', '_'))
        ofdir = os.path.dirname(ofpath)
        if not os.path.isdir(ofdir):
            os.makedirs(ofdir)
        h5f = h5py.File(ofpath, 'w')

        def store_person_predictions():
            global current_person_data
            if len(current_person_data) > 0:
                g = h5f.create_group(current_person_id)
                for key, data in current_person_data.items():
                    g.create_dataset(key, data=data, dtype=np.float32)
            current_person_data = {}
        with torch.no_grad():
            np.random.seed()
            num_batches = int(np.ceil(len(data_dict['dataset']) / args.eval_batch_size))
            for i, input_dict in enumerate(data_dict['dataloader']):
                # Get embeddings
                network.eval()
                output_dict = network(send_data_dict_to_gpu(input_dict))
                output_dict = dict([(k, v.cpu().numpy()) for k, v in output_dict.items()])

                # Process output line by line
                zipped_data = zip(
                    input_dict['key'],
                    input_dict['gaze_a'].cpu().numpy(),
                    input_dict['head_a'].cpu().numpy(),
                    output_dict['z_app'],
                    output_dict['z_gaze_enc'],
                    output_dict['z_head_enc'],
                    output_dict['gaze_a_hat'],
                )
                for (person_id, gaze, head, z_app, z_gaze, z_head, gaze_hat) in zipped_data:
                    # Store predictions if moved on to next person
                    if person_id != current_person_id:
                        store_person_predictions()
                        current_person_id = person_id
                    # Now write it
                    to_write = {
                        'gaze': gaze,
                        'head': head,
                        'z_app': z_app,
                        'z_gaze': z_gaze,
                        'z_head': z_head,
                        'gaze_hat': gaze_hat,
                    }
                    for k, v in to_write.items():
                        if k not in current_person_data:
                            current_person_data[k] = []
                        current_person_data[k].append(v.astype(np.float32))
                logging.info('[%s] processed batch [%04d/%04d] with %d entries.' %
                             (tag, i + 1, num_batches, len(next(iter(input_dict.values())))))
        store_person_predictions()
        logging.info('Completed processing %s' % tag)
    logging.info('Done')
