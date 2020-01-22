#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import cv2
import numpy as np
import random
import threading
import pickle
import sys

import torch
sys.path.append("../src")
from losses import GazeAngularLoss

directions = ['l', 'r', 'u', 'd']
keys = {'u': 82,
        'd': 84,
        'l': 81,
        'r': 83}

global THREAD_RUNNING
global frames

def create_image(mon, direction, i, color, target='E', grid=True, total=9):

    h = mon.h_pixels
    w = mon.w_pixels
    if grid:
        if total == 9:
            row = i % 3
            col = int(i / 3)
            x = int((0.02 + 0.48 * row) * w)
            y = int((0.02 + 0.48 * col) * h)
        elif total == 16:
            row = i % 4
            col = int(i / 4)
            x = int((0.05 + 0.3 * row) * w)
            y = int((0.05 + 0.3 * col) * h)
    else:
        x = int(random.uniform(0, 1) * w)
        y = int(random.uniform(0, 1) * h)

    # compute the ground truth point of regard
    x_cam, y_cam, z_cam = mon.monitor_to_camera(x, y)
    g_t = (x_cam, y_cam)

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = np.ones((h, w, 3), np.float32)
    if direction == 'r' or direction == 'l':
        if direction == 'r':
            cv2.putText(img, target, (x, y), font, 0.5, color, 2, cv2.LINE_AA)
        elif direction == 'l':
            cv2.putText(img, target, (w - x, y), font, 0.5, color, 2, cv2.LINE_AA)
            img = cv2.flip(img, 1)
    elif direction == 'u' or direction == 'd':
        imgT = np.ones((w, h, 3), np.float32)
        if direction == 'd':
            cv2.putText(imgT, target, (y, x), font, 0.5, color, 2, cv2.LINE_AA)
        elif direction == 'u':
            cv2.putText(imgT, target, (h - y, x), font, 0.5, color, 2, cv2.LINE_AA)
            imgT = cv2.flip(imgT, 1)
        img = imgT.transpose((1, 0, 2))

    return img, g_t


def grab_img(cap):
    global THREAD_RUNNING
    global frames
    while THREAD_RUNNING:
        _, frame = cap.read()
        frames.append(frame)


def collect_data(cap, mon, calib_points=9, rand_points=5):
    global THREAD_RUNNING
    global frames

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    calib_data = {'frames': [], 'g_t': []}

    i = 0
    while i < calib_points:

        # Start the sub-thread, which is responsible for grabbing images
        frames = []
        THREAD_RUNNING = True
        th = threading.Thread(target=grab_img, args=(cap,))
        th.start()
        direction = random.choice(directions)
        img, g_t = create_image(mon, direction, i, (0, 0, 0), grid=True, total=calib_points)
        cv2.imshow('image', img)
        key_press = cv2.waitKey(0)
        if key_press == keys[direction]:
            THREAD_RUNNING = False
            th.join()
            calib_data['frames'].append(frames)
            calib_data['g_t'].append(g_t)
            i += 1
        elif key_press & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        else:
            THREAD_RUNNING = False
            th.join()

    i = 0
    while i < rand_points:

        # Start the sub-thread, which is responsible for grabbing images
        frames = []
        THREAD_RUNNING = True
        th = threading.Thread(target=grab_img, args=(cap,))
        th.start()
        direction = random.choice(directions)
        img, g_t = create_image(mon, direction, i, (0, 0, 0), grid=False, total=rand_points)
        cv2.imshow('image', img)
        key_press = cv2.waitKey(0)
        if key_press == keys[direction]:
            THREAD_RUNNING = False
            th.join()
            calib_data['frames'].append(frames)
            calib_data['g_t'].append(g_t)
            i += 1
        elif key_press & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        else:
            THREAD_RUNNING = False
            th.join()
    cv2.destroyAllWindows()

    return calib_data


def fine_tune(subject, data, frame_processor, mon, device, gaze_network, k, steps=1000, lr=1e-4, show=False):

    # collect person calibration data
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('%s_calib.avi' % subject, fourcc, 30.0, (640, 480))
    target = []
    for index, frames in enumerate(data['frames']):
        n = 0
        for i in range(len(frames) - 10, len(frames)):
            frame = frames[i]
            g_t = data['g_t'][index]
            target.append(g_t)
            out.write(frame)

            # # show
            # cv2.putText(frame, str(n),(20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,0,0), 3, cv2.LINE_AA)
            # cv2.imshow('img', frame)
            # cv2.waitKey(30)

            n += 1
    cv2.destroyAllWindows()
    out.release()
    fout = open('%s_calib_target.pkl' % subject, 'wb')
    pickle.dump(target, fout)
    fout.close()

    vid_cap = cv2.VideoCapture('%s_calib.avi' % subject)
    data = frame_processor.process(subject, vid_cap, mon, device, gaze_network, por_available=True, show=show)
    vid_cap.release()

    n = len(data['image_a'])
    assert n==130, "Face not detected correctly. Collect calibration data again."
    _, c, h, w = data['image_a'][0].shape
    img = np.zeros((n, c, h, w))
    gaze_a = np.zeros((n, 2))
    head_a = np.zeros((n, 2))
    R_gaze_a = np.zeros((n, 3, 3))
    R_head_a = np.zeros((n, 3, 3))
    for i in range(n):
        img[i, :, :, :] = data['image_a'][i]
        gaze_a[i, :] = data['gaze_a'][i]
        head_a[i, :] = data['head_a'][i]
        R_gaze_a[i, :, :] = data['R_gaze_a'][i]
        R_head_a[i, :, :] = data['R_head_a'][i]

    # create data subsets
    train_indices = []
    for i in range(0, k*10, 10):
        train_indices.append(random.sample(range(i, i + 10), 3))
    train_indices = sum(train_indices, [])

    valid_indices = []
    for i in range(k*10, n, 10):
        valid_indices.append(random.sample(range(i, i + 10), 1))
    valid_indices = sum(valid_indices, [])

    input_dict_train = {
        'image_a': img[train_indices, :, :, :],
        'gaze_a': gaze_a[train_indices, :],
        'head_a': head_a[train_indices, :],
        'R_gaze_a': R_gaze_a[train_indices, :, :],
        'R_head_a': R_head_a[train_indices, :, :],
    }

    input_dict_valid = {
        'image_a': img[valid_indices, :, :, :],
        'gaze_a': gaze_a[valid_indices, :],
        'head_a': head_a[valid_indices, :],
        'R_gaze_a': R_gaze_a[valid_indices, :, :],
        'R_head_a': R_head_a[valid_indices, :, :],
    }

    for d in (input_dict_train, input_dict_valid):
        for k, v in d.items():
            d[k] = torch.FloatTensor(v).to(device).detach()

    #############
    # Finetuning
    #################

    loss = GazeAngularLoss()
    optimizer = torch.optim.SGD(
        [p for n, p in gaze_network.named_parameters() if n.startswith('gaze')],
        lr=lr,
    )

    gaze_network.eval()
    output_dict = gaze_network(input_dict_valid)
    valid_loss = loss(input_dict_valid, output_dict).cpu()
    print('%04d> , Validation: %.2f' % (0, valid_loss.item()))

    for i in range(steps):
        # zero the parameter gradient
        gaze_network.train()
        optimizer.zero_grad()

        # forward + backward + optimize
        output_dict = gaze_network(input_dict_train)
        train_loss = loss(input_dict_train, output_dict)
        train_loss.backward()
        optimizer.step()

        if i % 100 == 99:
            gaze_network.eval()
            output_dict = gaze_network(input_dict_valid)
            valid_loss = loss(input_dict_valid, output_dict).cpu()
            print('%04d> Train: %.2f, Validation: %.2f' %
                  (i+1, train_loss.item(), valid_loss.item()))
    torch.save(gaze_network.state_dict(), '%s_gaze_network.pth.tar' % subject)
    torch.cuda.empty_cache()

    return gaze_network