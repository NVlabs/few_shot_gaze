#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello.
# --------------------------------------------------------


import argparse
import os
import pickle
import random
from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from tensorboardX import SummaryWriter
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"""
    Utility functions
"""


def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-6, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-6, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))
    similarity = np.clip(similarity, a_min=-1.0 + 1e-6, a_max=1.0 - 1e-6)

    return np.degrees(np.arccos(similarity))


def nn_angular_error(y, y_hat):
    sim = F.cosine_similarity(y, y_hat, eps=1e-6)
    sim = F.hardtanh(sim, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(sim) * (180 / np.pi)


def nn_mean_angular_loss(y, y_hat):
    return torch.mean(nn_angular_error(y, y_hat))


def nn_mean_asimilarity(y, y_hat):
    return torch.mean(1.0 - F.cosine_similarity(y, y_hat, eps=1e-6))


def pitchyaw_to_vector(pitchyaws):
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


"""
    Tasks class for grabbing training/testing samples
"""


class Tasks(object):
    def __init__(self, hdf_path, x_keys=['z_gaze']):

        # Select tasks for which min. 1000 entries exist
        self.data = h5py.File(hdf_path, 'r')
        previous_len = len(self.data.keys())
        self.selected_tasks = [k for k in self.data.keys()
                               if self.data[k + '/gaze'].len() > 1000]
        self.num_tasks = len(self.selected_tasks)

        # Now load in all data into memory for selected tasks
        self.processed_data = []
        for task in self.selected_tasks:
            num_entries = self.data[task + '/gaze'].len()
            xs = np.concatenate([
                np.array(self.data[task + '/' + key]).reshape(num_entries, -1)
                for key in x_keys
            ], axis=1)
            ys = pitchyaw_to_vector(np.array(self.data[task + '/gaze']).reshape(-1, 2))
            self.processed_data.append((xs, ys))
        print('Loaded %s (%d -> %d tasks)' % (os.path.basename(hdf_path),
                                              previous_len, self.num_tasks))

        # By default, we just sample disjoint sets from the entire given data
        self.all_indices = [list(range(len(entries[0])))
                            for entries in self.processed_data]
        self.train_indices = self.all_indices
        self.test_indices = self.all_indices

    def create_sample(self, task_index, indices):
        """Create a sample of a task for meta-learning.

        This consists of a x, y pair.
        """
        xs, ys = zip(*[(self.processed_data[task_index][0][i],
                        self.processed_data[task_index][1][i])
                       for i in indices])
        xs, ys = np.array(xs), np.array(ys)
        return (torch.Tensor(xs).to(device),
                torch.Tensor(ys).to(device))

    def sample(self, num_train=4, num_test=100):
        """Yields training and testing samples."""
        picked_task = random.randint(0, self.num_tasks - 1)
        return self.sample_for_task(picked_task, num_train=num_train,
                                    num_test=num_test)

    def sample_for_task(self, task, num_train=4, num_test=100):
        if self.train_indices[task] is self.test_indices[task]:
            # This is for meta-training and meta-validation
            indices = random.sample(self.all_indices[task], num_train + num_test)
            train_indices = indices[:num_train]
            test_indices = indices[-num_test:]
        else:
            # This is for meta-testing
            train_indices = random.sample(self.train_indices[task], num_train)
            test_indices = self.test_indices[task]
        return (self.create_sample(task, train_indices),
                self.create_sample(task, test_indices))


class TestTasks(Tasks):
    """Class for final testing (not testing within meta-learning."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_indices = [indices[:-500] for indices in self.all_indices]
        self.test_indices = [indices[-500:] for indices in self.all_indices]


"""
    Replacement classes for standard PyTorch Module and Linear.
"""


class ModifiableModule(nn.Module):
    def params(self):
        return [p for _, p in self.named_params()]

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self):
        subparams = []
        for name, mod in self.named_submodules():
            for subname, param in mod.named_params():
                subparams.append((name + '.' + subname, param))
        return self.named_leaves() + subparams

    def set_param(self, name, param, copy=False):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in self.named_submodules():
                if module_name == name:
                    mod.set_param(rest, param, copy=copy)
                    break
        else:
            if copy is True:
                setattr(self, name, V(param.data.clone(), requires_grad=True))
            else:
                assert hasattr(self, name)
                setattr(self, name, param)

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            self.set_param(name, param, copy=not same_var)


class GradLinear(ModifiableModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs).to(device)

        nn.init.normal_(ignore.weight.data, mean=0.0, std=np.sqrt(1. / args[0]))
        nn.init.constant_(ignore.bias.data, val=0)

        self.weights = V(ignore.weight.data, requires_grad=True).to(device)
        self.bias = V(ignore.bias.data, requires_grad=True).to(device)

    def forward(self, x):
        return F.linear(x, self.weights, self.bias).to(device)

    def named_leaves(self):
        return [('weights', self.weights), ('bias', self.bias)]


"""
    Meta-learnable fully-connected neural network model definition
"""


class GazeEstimationModel(ModifiableModule):
    def __init__(self, activation_type='relu', layer_num_features=[48, 64, 3], make_alpha=False):
        super().__init__()
        self.activation_type = activation_type

        # Construct layers
        self.layer_num_features = layer_num_features
        self.layers = []
        for i, f_now in enumerate(self.layer_num_features[:-1]):
            f_next = self.layer_num_features[i + 1]
            layer = GradLinear(f_now, f_next)
            self.layers.append(('layer%02d' % (i + 1), layer))

        # For use with Meta-SGD
        self.alphas = []
        if make_alpha:
            for i, f_now in enumerate(self.layer_num_features[:-1]):
                f_next = self.layer_num_features[i + 1]
                alphas = GradLinear(f_now, f_next)
                alphas.weights.data.uniform_(0.005, 0.1)
                alphas.bias.data.uniform_(0.005, 0.1)
                self.alphas.append(('alpha%02d' % (i + 1), alphas))

    def clone(self, make_alpha=None):
        if make_alpha is None:
            make_alpha = (self.alphas is not None and len(self.alphas) > 0)
        new_model = self.__class__(self.activation_type, self.layer_num_features,
                                   make_alpha=make_alpha)
        new_model.copy(self)
        return new_model

    def state_dict(self):
        output = {}
        for key, layer in self.layers:
            output[key + '.weights'] = layer.weights.data
            output[key + '.bias'] = layer.bias.data
        return output

    def load_state_dict(self, weights):
        for key, tensor in weights.items():
            self.set_param(key, tensor, copy=True)

    def forward(self, x):
        for name, layer in self.layers[:-1]:
            x = layer(x)
            if self.activation_type == 'relu':
                x = F.relu_(x)
            elif self.activation_type == 'leaky_relu':
                x = F.leaky_relu_(x)
            elif self.activation_type == 'elu':
                x = F.elu_(x)
            elif self.activation_type == 'selu':
                x = F.selu_(x)
            elif self.activation_type == 'tanh':
                x = torch.tanh_(x)
            elif self.activation_type == 'sigmoid':
                x = torch.sigmoid_(x)
            elif self.activation_type == 'none':
                pass
            else:
                raise ValueError('Unknown activation function "%s"' % self.activation_type)
        x = self.layers[-1][1](x)  # No activation on output of last layer
        x = F.normalize(x, dim=-1)  # Normalize
        return x

    def named_submodules(self):
        return self.layers + self.alphas


class GazeEstimationModelPreExtended(ModifiableModule):
    def __init__(self):
        super().__init__()

        # Construct layers
        self.layer00 = GradLinear(640, 118)  # 64 + 2*3 + 16*3
        self.layer01 = GradLinear(118, 64)
        self.layer02 = GradLinear(64, 3)
        self.layers = [('layer00', self.layer00),
                       ('layer01', self.layer01),
                       ('layer02', self.layer02)]

    def clone(self, make_alpha=None):
        new_model = self.__class__()
        new_model.copy(self)
        return new_model

    def forward(self, x):
        x = self.layer00(x)
        x = x[:, 64:70]  # Extract at hardcoded z_gaze indices
        x = F.selu_(x)
        x = self.layer01(x)
        x = F.selu_(x)
        x = self.layer02(x)
        x = F.normalize(x, dim=-1)  # Normalize
        return x

    def named_submodules(self):
        return self.layers


"""
    Meta-learning utility functions.
"""


def forward_and_backward(model, data, optim=None, create_graph=False,
                         train_data=None, loss_function=nn_mean_angular_loss):
    model.train()
    if optim is not None:
        optim.zero_grad()
    loss = forward(model, data, train_data=train_data, for_backward=True,
                   loss_function=loss_function)
    loss.backward(create_graph=create_graph, retain_graph=(optim is None))
    if optim is not None:
        # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
    return loss.data.cpu().numpy()


def forward(model, data, return_predictions=False, train_data=None,
            for_backward=False, loss_function=nn_mean_angular_loss):
    model.train()
    x, y = data
    y_hat = model(V(x))
    loss = loss_function(y_hat, V(y))
    if return_predictions:
        return y_hat.data.cpu().numpy()
    elif for_backward:
        return loss
    else:
        return loss.data.cpu().numpy()


"""
    Inference through model (with/without gradient calculation)
"""


class MAML(object):
    def __init__(self, model, k, output_dir='./outputs/',
                 train_tasks=None, valid_tasks=None, no_tensorboard=False):
        self.model = model
        self.meta_model = model.clone()

        self.train_tasks = train_tasks
        self.valid_tasks = valid_tasks
        self.k = k

        self.output_dir = None
        self.tensorboard = None
        if output_dir is not None:
            self.output_dir = '%s/%s_%02d' % (output_dir, self.__class__.__name__, k)
            if not os.path.isdir(self.output_dir):
                os.makedirs(self.output_dir)

            if not no_tensorboard:
                self.tensorboard = SummaryWriter(self.output_dir)

    @property
    def model_parameters_path(self):
        return '%s/meta_learned_parameters.pth.tar' % self.output_dir

    def save_model_parameters(self):
        if self.output_dir is not None:
            torch.save(self.model.state_dict(), self.model_parameters_path)

    def load_model_parameters(self):
        if os.path.isfile(self.model_parameters_path):
            weights = torch.load(self.model_parameters_path)
            self.model.load_state_dict(weights)
            print('> Loaded weights from %s' % self.model_parameters_path)

    def train(self, steps_outer, steps_inner=1, lr_inner=0.01, lr_outer=0.001,
              disable_tqdm=False):
        self.lr_inner = lr_inner
        print('\nBeginning meta-learning for k = %d' % self.k)
        print('> Please check tensorboard logs for progress.\n')

        # Outer loop optimizer
        optimizer = torch.optim.Adam(self.model.params(), lr=lr_outer)

        # Model and optimizer for validation
        valid_model = self.model.clone()
        valid_optim = torch.optim.SGD(valid_model.params(), lr=self.lr_inner)

        for i in tqdm(range(steps_outer), disable=disable_tqdm):
            for j in range(steps_inner):
                # Make copy of main model
                self.meta_model.copy(self.model, same_var=True)

                # Get a task
                train_data, test_data = self.train_tasks.sample(num_train=self.k)

                # Run the rest of the inner loop
                task_loss = self.inner_loop(train_data, self.lr_inner)

            # Calculate gradients on a held-out set
            new_task_loss = forward_and_backward(
                self.meta_model, test_data, train_data=train_data,
            )

            # Update the main model
            optimizer.step()
            optimizer.zero_grad()

            if (i + 1) % 100 == 0:
                # Log to Tensorflow
                if self.tensorboard is not None:
                    self.tensorboard.add_scalar('meta-train/train-loss', task_loss, i)
                    self.tensorboard.add_scalar('meta-train/valid-loss', new_task_loss, i)

                # Validation
                losses = []
                for j in range(self.valid_tasks.num_tasks):
                    valid_model.copy(self.model)
                    train_data, test_data = self.valid_tasks.sample_for_task(j, num_train=self.k)
                    train_loss = forward_and_backward(valid_model, train_data, valid_optim)
                    valid_loss = forward(valid_model, test_data, train_data=train_data)
                    losses.append((train_loss, valid_loss))
                train_losses, valid_losses = zip(*losses)
                if self.tensorboard is not None:
                    self.tensorboard.add_scalar('meta-valid/train-loss', np.mean(train_losses), i)
                    self.tensorboard.add_scalar('meta-valid/valid-loss', np.mean(valid_losses), i)

        # Save MAML initial parameters
        self.save_model_parameters()

    def test(self, test_tasks_list, num_iterations=[1, 5, 10], num_repeats=20):
        print('\nBeginning testing for meta-learned model with k = %d\n' % self.k)
        model = self.model.clone()

        # IMPORTANT
        #
        # Sets consistent seed such that as long as --num-test-repeats is the
        # same, experiment results from multiple invocations of this script can
        # yield the same calibration samples.
        random.seed(4089213955)

        for test_set_name, test_tasks in test_tasks_list.items():
            predictions = OrderedDict()
            losses = OrderedDict([(n, []) for n in num_iterations])
            for i, task_name in enumerate(test_tasks.selected_tasks):
                predictions[task_name] = []
                for t in range(num_repeats):
                    model.copy(self.model)
                    optim = torch.optim.SGD(model.params(), lr=self.lr_inner)

                    train_data, test_data = test_tasks.sample_for_task(i, num_train=self.k)
                    if num_iterations[0] == 0:
                        train_loss = forward(model, train_data)
                        test_loss = forward(model, test_data, train_data=train_data)
                        losses[0].append((train_loss, test_loss))
                    for j in range(np.amax(num_iterations)):
                        train_loss = forward_and_backward(model, train_data, optim)
                        if (j + 1) in num_iterations:
                            test_loss = forward(model, test_data, train_data=train_data)
                            losses[j + 1].append((train_loss, test_loss))

                    # Register ground truth and prediction
                    predictions[task_name].append({
                        'groundtruth': test_data[1].cpu().numpy(),
                        'predictions': forward(model, test_data,
                                               return_predictions=True,
                                               train_data=train_data),
                    })
                    predictions[task_name][-1]['errors'] = angular_error(
                        predictions[task_name][-1]['groundtruth'],
                        predictions[task_name][-1]['predictions'],
                    )

                print('Done for k = %3d, %s/%s... train: %.3f, test: %.3f' % (
                    self.k, test_set_name, task_name,
                    np.mean([both[0] for both in losses[num_iterations[-1]][-num_repeats:]]),
                    np.mean([both[1] for both in losses[num_iterations[-1]][-num_repeats:]]),
                ))

            if self.output_dir is not None:
                # Save predictions to file
                pkl_path = '%s/predictions_%s.pkl' % (self.output_dir, test_set_name)
                with open(pkl_path, 'wb') as f:
                    pickle.dump(predictions, f)

                # Finally, log values to tensorboard
                if self.tensorboard is not None:
                    for n, v in losses.items():
                        train_losses, test_losses = zip(*v)
                        stem = 'meta-test/%s/' % test_set_name
                        self.tensorboard.add_scalar(stem + 'train-loss', np.mean(train_losses), n)
                        self.tensorboard.add_scalar(stem + 'valid-loss', np.mean(test_losses), n)

                # Write loss values as plain text too
                np.savetxt('%s/losses_%s_train.txt' % (self.output_dir, test_set_name),
                           [[n, np.mean(list(zip(*v))[0])] for n, v in losses.items()])
                np.savetxt('%s/losses_%s_valid.txt' % (self.output_dir, test_set_name),
                           [[n, np.mean(list(zip(*v))[1])] for n, v in losses.items()])

            out_msg = '> Completed test on %s for k = %d' % (test_set_name, self.k)
            final_n = sorted(num_iterations)[-1]
            final_train_losses, final_test_losses = zip(*(losses[final_n]))
            out_msg += ('\n  at %d steps losses were... train: %.3f, test: %.3f +/- %.3f' %
                        (final_n, np.mean(final_train_losses),
                         np.mean(final_test_losses),
                         np.mean([
                             np.std([
                                 data['errors'] for data in person_data
                             ], axis=0)
                             for person_data in predictions.values()
                         ])))
            print(out_msg)

    def inner_loop(self, train_data, lr_inner=0.01):
        # Forward-pass and calculate gradients on meta model
        loss = forward_and_backward(self.meta_model, train_data,
                                    create_graph=True)

        # Apply gradients
        for name, param in self.meta_model.named_params():
            self.meta_model.set_param(name, param - lr_inner * param.grad)
        return loss


class FOMAML(MAML):
    def inner_loop(self, train_data, lr_inner=0.01):
        # Forward-pass and calculate gradients on meta model
        loss = forward_and_backward(self.meta_model, train_data)

        # Apply gradients
        for name, param in self.meta_model.named_params():
            grad = V(param.grad.detach().data)
            self.meta_model.set_param(name, param - lr_inner * grad)
        return loss


class MetaSGD(MAML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model.clone(make_alpha=True)
        self.meta_model = self.model.clone()

    def inner_loop(self, train_data, lr_inner=0.01):
        # Forward-pass and calculate gradients on meta model
        loss = forward_and_backward(self.meta_model, train_data,
                                    create_graph=True)

        # Apply gradients
        named_params = dict(self.meta_model.named_params())
        for name, param in named_params.items():
            if name.startswith('layer'):
                alpha = named_params['alpha' + str(name[5:])]
                self.meta_model.set_param(name, param - lr_inner * alpha * param.grad)
        return loss


class NONE(MAML):
    def train(self, steps_outer, steps_inner=1, lr_inner=0.01, lr_outer=0.001,
              disable_tqdm=False):
        self.lr_inner = lr_inner

        # Save randomly initialized MLP parameters
        self.save_model_parameters()


"""
    Actual run script
"""

if __name__ == '__main__':

    # Available meta-learning methods
    meta_learner_classes = {
        'MAML': MAML,
        'FOMAML': FOMAML,
        'Meta-SGD': MetaSGD,
        'NONE': NONE,
    }

    # Define and parse configuration for training and evaluations
    parser = argparse.ArgumentParser(description='Meta-learn gaze estimator from RotAE embeddings.')
    parser.add_argument('input_dir', type=str,
                        help='Input directory for experiment data')
    parser.add_argument('--output-dir', type=str, default='./',
                        help='Output directory for tensorboard log relative to input dir')
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='Log training and validation progress to tensorboard.')
    parser.add_argument('--disable-tqdm', action='store_true',
                        help='Disable progress bar from tqdm (in particular on NGC).')

    parser.add_argument('--maml-use-pretrained-mlp', action='store_true',
                        help='Even for MAML, use pre-trained MLP paramters.')

    # Gaze estimation neural network configuration
    parser.add_argument('--select-z', type=str, default='z_gaze',
                        help='Embeddings/features to select for using as input to MAML '
                             + '(default: z_gaze)')
    parser.add_argument('--layer-num-features', type=str, default='64',
                        help='Network configuration, number of FC features delimited by \',\' '
                             + '(default: 64)')
    parser.add_argument('--activation', type=str, default='selu',
                        choices=['sigmoid', 'relu', 'leaky_relu', 'elu', 'selu', 'tanh', 'none'],
                        help='Neural network activation function.')

    # Parameters for meta-learning
    parser.add_argument('--meta-learner', type=str, default='MAML',
                        choices=list(meta_learner_classes.keys()),
                        help='Meta-learning algorithm')
    parser.add_argument('--steps-meta-training', type=int, default=100000,
                        help='Number of steps to meta-learn for (default: 100000)')
    parser.add_argument('--tasks-per-meta-iteration', type=int, default=5,
                        help='Tasks to evaluate per meta-learning iteration (default: 5)')
    parser.add_argument('--lr-inner', type=float, default=1e-5,
                        help='Learning rate for inner loop (for the task) (default: 1e-5)')
    parser.add_argument('--lr-outer', type=float, default=1e-3,
                        help='Learning rate for outer loop (the meta learner) (default: 1e-3)')

    # Evaluation
    parser.add_argument('--skip-training', action='store_true',
                        help='Skips meta-training')
    parser.add_argument('k', type=int,
                        help='Number of calibration samples to use - k as in k-shot learning.')
    parser.add_argument('--num-test-repeats', type=int, default=10,
                        help='Number of times to repeat drawing of k samples for testing '
                             + '(default: 10)')
    parser.add_argument('--steps-testing', type=int, default=1000,
                        help='Number of steps to meta-learn for (default: 1000)')

    args = parser.parse_args()

    # Define data sources (tasks)
    x_keys = args.select_z.split(',')
    meta_train_tasks = Tasks(args.input_dir + '/gc_train_predictions.h5', x_keys=x_keys)
    meta_val_tasks = Tasks(args.input_dir + '/gc_val_predictions.h5', x_keys=x_keys)
    meta_test_tasks = [
        ('gc', TestTasks(args.input_dir + '/gc_test_predictions.h5', x_keys=x_keys)),
        ('mpi', TestTasks(args.input_dir + '/mpi_predictions.h5', x_keys=x_keys)),
    ]

    # Construct output directory path string
    output_dir = None
    if args.output_dir is not None:
        output_dir = (os.path.realpath(args.input_dir + '/' + args.output_dir)
                      if args.output_dir[0] != '/' else args.output_dir)
        output_dir += '/'
        output_dir += 'Zg'
        output_dir += '_OLR%.0e' % args.lr_outer
        output_dir += '_IN%d' % args.tasks_per_meta_iteration
        output_dir += '_ILR%.0e' % args.lr_inner
        # output_dir += '_OutN%e' % args.steps_meta_training
        output_dir += '_Net%s' % args.layer_num_features.replace(',', '-')

    # Get an example entry to design gaze estimation model
    sample_train, _ = meta_train_tasks.sample(num_train=1, num_test=0)

    # Training configuration
    layer_num_features = [int(f) for f in args.layer_num_features.split(',')]
    layer_num_features = [sample_train[0].shape[1]] + layer_num_features + [3]
    if not args.select_z == 'before_z':
        model = GazeEstimationModel(activation_type=args.activation,
                                    layer_num_features=layer_num_features)
    else:
        assert args.maml_use_pretrained_mlp is True
        assert args.layer_num_features == '64'
        assert args.activation == 'selu'
        model = GazeEstimationModelPreExtended()
    meta_learner_class = meta_learner_classes[args.meta_learner]
    meta_learner = meta_learner_class(model, args.k, output_dir,
                                      meta_train_tasks, meta_val_tasks,
                                      no_tensorboard=args.no_tensorboard)

    # If doing fine-tuning... try to load pre-trained MLP weights
    if args.meta_learner == 'NONE' or args.maml_use_pretrained_mlp:
        import glob
        checkpoint_path = sorted(
            glob.glob('%s/checkpoints/at_step_*.pth.tar' % args.input_dir)
        )[-1]
        weights = torch.load(checkpoint_path)
        try:
            state_dict = {
                'layer01.weights': weights['module.gaze1.weight'],
                'layer01.bias': weights['module.gaze1.bias'],
                'layer02.weights': weights['module.gaze2.weight'],
                'layer02.bias': weights['module.gaze2.bias'],
            }
            if args.select_z == 'before_z':
                state_dict['layer00.weights'] = weights['module.fc_enc.weight']
                state_dict['layer00.bias'] = weights['module.fc_enc.bias']
        except:  # noqa
            state_dict = {
                'layer01.weights': weights['gaze1.weight'],
                'layer01.bias': weights['gaze1.bias'],
                'layer02.weights': weights['gaze2.weight'],
                'layer02.bias': weights['gaze2.bias'],
            }
            if args.select_z == 'before_z':
                state_dict['layer00.weights'] = weights['fc_enc.weight']
                state_dict['layer00.bias'] = weights['fc_enc.bias']
        for key, values in state_dict.items():
            model.set_param(key, values, copy=True)
        del state_dict
        print('Loaded %s' % checkpoint_path)

    if not args.skip_training:
        meta_learner.train(
            steps_outer=args.steps_meta_training,
            steps_inner=args.tasks_per_meta_iteration,
            lr_inner=args.lr_inner,
            lr_outer=args.lr_outer,
            disable_tqdm=args.disable_tqdm,
        )

    # Perform test (which entails the repeated training of person-specific models
    if args.skip_training:
        meta_learner.load_model_parameters()
    meta_learner.lr_inner = args.lr_inner
    meta_learner.test(
        test_tasks_list=OrderedDict(meta_test_tasks),
        num_iterations=list(np.arange(start=0, stop=args.steps_testing + 1, step=20)),
        num_repeats=args.num_test_repeats,
    )
