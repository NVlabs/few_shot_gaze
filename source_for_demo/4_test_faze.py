#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello.
# --------------------------------------------------------

import os
import pickle
import cv2 as cv
import sys

import numpy as np
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F

#################
# Configurations
#################
ted_parameters_path      = './weights_ted.pth.tar' #'../src/outputs_of_full_train_test_and_plot/checkpoints/at_step_0057101.pth.tar'
maml_parameters_path     = './weights_maml' #'../src/outputs_of_full_train_test_and_plot/Zg_OLR1e-03_IN5_ILR1e-05_Net64'
num_finetuning_steps     = 1000
finetuning_learning_rate = 5e-4
k                        = 9

##############
# Setup model

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create network
# WARNING/NOTE: MAKE SURE THIS IS ABSOLUTELY EXACTLY THE SETTING IN WHICH THE
#               PREDICTIONS WERE GENERATED IN.
# Create network
sys.path.append("../src")
from models import DTED
network = DTED(
    growth_rate=32,
    z_dim_app=64,
    z_dim_gaze=2,
    z_dim_head=16,
    decoder_input_c=32,
    normalize_3d_codes=True,
    normalize_3d_codes_axis=1,
    backprop_gaze_to_encoder=False,
).to(device)

#################################
# Load T-ED weights if available

assert os.path.isfile(ted_parameters_path)
print('> Loading: %s' % ted_parameters_path)
ted_weights = torch.load(ted_parameters_path)
if torch.cuda.device_count() == 1:
    if next(iter(ted_weights.keys())).startswith('module.'):
        ted_weights = dict([(k[7:], v) for k, v in ted_weights.items()])

#####################################
# Load MAML MLP weights if available

full_maml_parameters_path = maml_parameters_path + '/%02d.pth.tar' % k  #'/MAML_%02d/meta_learned_parameters.pth.tar' % k
assert os.path.isfile(full_maml_parameters_path)
print('> Loading: %s' % full_maml_parameters_path)
maml_weights = torch.load(full_maml_parameters_path)
ted_weights.update({  # rename to fit
    'gaze1.weight': maml_weights['layer01.weights'],
    'gaze1.bias':   maml_weights['layer01.bias'],
    'gaze2.weight': maml_weights['layer02.weights'],
    'gaze2.bias':   maml_weights['layer02.bias'],
})
network.load_state_dict(ted_weights)

#####################
# Example input data

with open('./sample_person_data.pkl', 'rb') as f:
    data = pickle.load(f)

    def preprocess_image(image):
        ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv.equalizeHist(ycrcb[:, :, 0])
        image = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB)
        image = np.transpose(image, [2, 0, 1])  # Colour image
        image = 2.0 * image / 255.0 - 1
        return image

    # Functions to calculate relative rotation matrices for gaze dir. and head pose
    def R_x(theta):
        sin_ = np.sin(theta)
        cos_ = np.cos(theta)
        return np.array([
            [1., 0., 0.],
            [0., cos_, -sin_],
            [0., sin_, cos_]
        ]).astype(np.float32)

    def R_y(phi):
        sin_ = np.sin(phi)
        cos_ = np.cos(phi)
        return np.array([
            [cos_, 0., sin_],
            [0., 1., 0.],
            [-sin_, 0., cos_]
        ]).astype(np.float32)

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


    g_a = pitchyaw_to_vector(data['labels'][:, :2])
    h_a = pitchyaw_to_vector(data['labels'][:, 2:4])

    n, h, w, c = data['pixels'].shape
    img = np.zeros((n, c, h, w))
    R_gaze_a = np.zeros((n, 3, 3))
    R_head_a = np.zeros((n, 3, 3))
    for i in range(data['pixels'].shape[0]):
        img[i, :, :, :] = preprocess_image(data['pixels'][i, :, :, :])
        R_gaze_a[i, :, :] = calculate_rotation_matrix(g_a[i, :])
        R_head_a[i, :, :] = calculate_rotation_matrix(h_a[i, :])

    # reduce the number of validation samples if
    # you have less GPU memory
    num_indices = img.shape[0]
    train_indices = np.random.permutation(num_indices-200)[:k]
    valid_indices = np.arange(num_indices-200, num_indices)

    input_dict_train = {
        'image_a': img[train_indices, :, :, :],
        'gaze_a': g_a[train_indices, :],
        'head_a': h_a[train_indices, :],
        'R_gaze_a': R_gaze_a[train_indices, :, :],
        'R_head_a': R_head_a[train_indices, :, :],
    }

    input_dict_valid = {
        'image_a': img[valid_indices, :, :, :],
        'gaze_a': g_a[valid_indices, :],
        'head_a': h_a[valid_indices, :],
        'R_gaze_a': R_gaze_a[valid_indices, :, :],
        'R_head_a': R_head_a[valid_indices, :, :],
    }

    for d in (input_dict_train, input_dict_valid):
        for k, v in d.items():
            d[k] = torch.FloatTensor(v).to(device).detach()

#############
# Finetuning

def nn_angular_distance(a, b):
    sim = F.cosine_similarity(a, b, eps=1e-6)
    sim = F.hardtanh(sim, 1e-6, 1.0 - 1e-6)
    dist = torch.acos(sim) * (180 / np.pi)
    return torch.mean(dist)

optimizer = torch.optim.SGD(
    [p for n, p in network.named_parameters() if n.startswith('gaze')],
    lr=finetuning_learning_rate,
)

network.eval()
output_dict = network(input_dict_valid)
valid_loss = nn_angular_distance(output_dict['gaze_a_hat'], input_dict_valid['gaze_a'])
print('%04d> , Validation: %.2f' % (0, valid_loss.item()))

for i in range(num_finetuning_steps):
    # zero the parameter gradient
    network.train()
    optimizer.zero_grad()

    # forward + backward + optimize
    output_dict = network(input_dict_train)
    train_loss = nn_angular_distance(output_dict['gaze_a_hat'], input_dict_train['gaze_a'])
    train_loss.backward()
    optimizer.step()

    if i % 100 == 99:
        network.eval()
        output_dict = network(input_dict_valid)
        valid_loss = nn_angular_distance(output_dict['gaze_a_hat'], input_dict_valid['gaze_a'])
        print('%04d> Train: %.2f, Validation: %.2f' %
              (i+1, train_loss.item(), valid_loss.item()))
