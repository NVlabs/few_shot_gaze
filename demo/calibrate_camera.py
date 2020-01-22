#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

from subprocess import call
import cv2
import numpy as np
from os import path

from camera import cam_calibrate


#################################
# Start camera
#################################

cam_idx = 0

# adjust these for your camera to get the best accuracy
# use the same parameters to run the actual demoqsqq
call('v4l2-ctl -d /dev/video%d -c brightness=100' % cam_idx, shell=True)
call('v4l2-ctl -d /dev/video%d -c contrast=50' % cam_idx, shell=True)
call('v4l2-ctl -d /dev/video%d -c sharpness=100' % cam_idx, shell=True)

cam_cap = cv2.VideoCapture(cam_idx)
cam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# calibrate camera
cam_calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}
print("Calibrate camera once. Print pattern.png, paste on a clipboard, show to camera and capture non-blurry images in which points are detected well.")
print("Press s to save frame, c to continue to next frame and q to quit collecting data and proceed to calibration.")
cam_calibrate(cam_idx, cam_cap, cam_calib)
cam_cap.release()