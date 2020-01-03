#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello.
# --------------------------------------------------------

import cv2
from subprocess import call
import numpy as np
from os import path
import pickle
import sys
import torch
import os

from undistorter import Undistorter
from KalmanFilter1D import Kalman1D

from face import face
from landmarks import landmarks
from head import PnPHeadPoseEstimator
from normalization import normalize


#################
# Configurations
#################
ted_parameters_path = '../source_for_demo/weights_ted.pth.tar'  #'../src/outputs_of_full_train_test_and_plot/checkpoints/at_step_0057101.pth.tar'
maml_parameters_path = '../source_for_demo/weights_maml'  #'../src/outputs_of_full_train_test_and_plot/Zg_OLR1e-03_IN5_ILR1e-05_Net64'
k = 9

##############
# Setup model

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create network
sys.path.append("../src")
from models import DTED
gaze_network = DTED(
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

full_maml_parameters_path = maml_parameters_path +'/%02d.pth.tar' % k #maml_parameters_path +'/MAML_%02d/meta_learned_parameters.pth.tar' % k
assert os.path.isfile(full_maml_parameters_path)
print('> Loading: %s' % full_maml_parameters_path)
maml_weights = torch.load(full_maml_parameters_path)
ted_weights.update({  # rename to fit
    'gaze1.weight': maml_weights['layer01.weights'],
    'gaze1.bias':   maml_weights['layer01.bias'],
    'gaze2.weight': maml_weights['layer02.weights'],
    'gaze2.bias':   maml_weights['layer02.bias'],
})
gaze_network.load_state_dict(ted_weights)


# get monitor resolution:
from screeninfo import get_monitors
m = get_monitors()
def parseDimensions(monitorString):
    delim1 = str.find(monitorString, '(')
    delim2 = str.find(monitorString, 'x')
    delim3 = str.find(monitorString, '+')
    w = int(monitorString[delim1 + 1:delim2])
    h = int(monitorString[delim2 + 1:delim3])

    return h, w
mon_h, mon_w = parseDimensions(str(m))

def cam_calibrate(cam_idx, cap, cam_calib):

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    pts = np.zeros((6 * 9, 3), np.float32)
    pts[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # capture calibration frames
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.
    frames = []
    while True:
        ret, frame = cap.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            if ret:
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                # Draw and display the corners
                frame_copy = frame.copy()
                cv2.drawChessboardCorners(frame_copy, (9, 6), corners, ret)
                cv2.imshow('points', frame_copy)

                # s to save, c to continue, q to quit
                if cv2.waitKey(0) & 0xFF == ord('s'):
                    img_points.append(corners)
                    obj_points.append(pts)
                    frames.append(frame)
                elif cv2.waitKey(0) & 0xFF == ord('n'):
                    continue
                elif cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

    # compute calibration matrices
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, frames[0].shape[0:2], None, None)

    # check
    error = 0.0
    for i in range(len(frames)):
        proj_imgpoints, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error += (cv2.norm(img_points[i], proj_imgpoints, cv2.NORM_L2) / len(proj_imgpoints))
    print("Camera calibrated successfully, total re-projection error: %f" % (error / len(frames)))

    cam_calib['mtx'] = mtx
    cam_calib['dist'] = dist
    print("Camera parameters:")
    print(cam_calib)

    pickle.dump(cam_calib, open("calib_%d.pkl" % (cam_idx), "wb"))




# adjust these parameters for your webcam to obtain the best results
call("v4l2-ctl -d /dev/video0 -c brightness=75", shell=True)
call("v4l2-ctl -d /dev/video0 -c contrast=50", shell=True)
call("v4l2-ctl -d /dev/video0 -c sharpness=100", shell=True)

cam_idx = 0
cap = cv2.VideoCapture(cam_idx)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# read the camera matrix and create an
cam_calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}
if path.exists("calib_%d.pkl" % (cam_idx)):
    cam_calib = pickle.load(open("calib_%d.pkl" % (cam_idx), "rb"))
else:
    print("Calibrate camera once. Print pattern.png, paste on a clipboard, show to camera and capture non-blurry images in which points are detected well.")
    print("Press s to save frame, c to continue, q to quit")
    cam_calibrate(cam_idx, cap, cam_calib)

undistorter = Undistorter(cam_calib['mtx'], cam_calib['dist'])

#######################################################
#### prepare Kalman filters, R can change behaviour of Kalman filter
#### play with it to get better smoothing, larger R - more smoothing and larger delay
#######################################################
kalman_filters = list()
for point in range(2):
    #initialize kalman filters for different coordinates
    #will be used for face detection over a single object
    kalman_filters.append(Kalman1D(sz=100, R=0.0075**2))

kalman_filters_landm = list()
for point in range(68):
    #initialize kalman filters for different coordinates
    #will be used to smooth landmarks over the face for a single face tracking
    kalman_filters_landm.append(Kalman1D(sz=100, R=0.001**2))

# initialize kalman filters for different coordinates
# will be used to smooth landmarks over the face for a single face tracking
kalman_filter_gaze = list()
kalman_filter_gaze.append(Kalman1D(sz=100, R=0.005 ** 2))

landmarks_detector = landmarks()
head_pose_estimator = PnPHeadPoseEstimator()

def equalize(image):  # Proper colour image intensity equalization
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return output

ret, img = cap.read()
img = undistorter.apply(img)
while ret:
    ret, img = cap.read()
    img = undistorter.apply(img)

    # detect face
    face_location = face.detect(img,  use_max='SIZE')

    if len(face_location) > 0:
        ##use kalman filter to smooth bounding box position
        ##assume work with complex numbers:
        output_tracked = kalman_filters[0].update(face_location[0] + 1j * face_location[1])
        face_location[0], face_location[1] = np.real(output_tracked), np.imag(output_tracked)
        output_tracked = kalman_filters[1].update(face_location[2] + 1j * face_location[3])
        face_location[2], face_location[3] = np.real(output_tracked), np.imag(output_tracked)

        # detect facial points
        pts = landmarks_detector.detect(face_location, img)
        ##run Kalman filter on landmarks to smooth them
        for i in range(68):
            kalman_filters_landm_complex = kalman_filters_landm[i].update(pts[i, 0] + 1j * pts[i, 1])
            pts[i, 0], pts[i, 1] = np.real(kalman_filters_landm_complex), np.imag(kalman_filters_landm_complex)

        # compute head pose
        fx, _, cx, _, fy, cy, _, _, _ = cam_calib['mtx'].flatten()
        camera_parameters = np.asarray([fx, fy, cx, cy])
        rvec, tvec = head_pose_estimator.fit_func(pts, camera_parameters)

        ######### GAZE PART #########

        # create eye patch
        head_pose = (rvec, tvec)
        por = None
        entry = {
                'full_frame': img,
                '3d_gaze_target': por,
                'camera_parameters': camera_parameters,
                'full_frame_size': (img.shape[0], img.shape[1]),
                'face_bounding_box': (int(face_location[0]), int(face_location[1]),
                                      int(face_location[2] - face_location[0]),
                                      int(face_location[3] - face_location[1]))
                }
        [patch, h_n, g_n, inverse_M, gaze_cam_origin, gaze_cam_target] = normalize(entry, head_pose)
        cv2.imshow('raw patch', patch)

        def preprocess_image(image):
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
            cv2.imshow('processed patch', image)

            image = np.transpose(image, [2, 0, 1])  # CxHxW
            image = 2.0 * image / 255.0 - 1
            return image

        # estimate the PoR using the gaze network
        processed_patch = preprocess_image(patch)
        processed_patch = processed_patch[np.newaxis, :, :, :]
        input_dict = {
            'image_a': processed_patch,
            'gaze_a': np.zeros((1, 1, 2)),
            'head_a': np.zeros((1, 1, 2)),
            'R_gaze_a': np.zeros((1, 3, 3)),
            'R_head_a': np.zeros((1, 3, 3)),
        }
        # compute eye gaze and point of regard
        for k, v in input_dict.items():
            input_dict[k] = torch.FloatTensor(v).to(device).detach()

        gaze_network.eval()
        output_dict = gaze_network(input_dict)
        output = output_dict['gaze_a_hat']
        g_cnn = output.data.cpu().numpy()
        g_cnn = g_cnn.reshape(3,1)
        g_cnn /= np.linalg.norm(g_cnn)

        # compute the POR on z=0 plane
        g_n_forward = -g_cnn
        g_cam_forward = inverse_M * g_n_forward
        g_cam_forward = g_cam_forward / np.linalg.norm(g_cam_forward)

        d = -gaze_cam_origin[2] / g_cam_forward[2]
        por_cam_x = gaze_cam_origin[0] + d * g_cam_forward[0]
        por_cam_y = gaze_cam_origin[1] + d * g_cam_forward[1]
        por_cam_z = 0.0

        x_pixel_hat = np.ceil(960 - por_cam_x * 1920 / 345.4)
        y_pixel_hat = np.ceil((por_cam_y - 10) * 1080 / 194.1)

        output_tracked = kalman_filter_gaze[0].update(x_pixel_hat + 1j * y_pixel_hat)
        x_pixel_hat, y_pixel_hat = np.ceil(np.real(output_tracked)), np.ceil(np.imag(output_tracked))

        # show point of regard on screen
        display = np.ones((mon_h, mon_w, 3), np.float32)
        h, w, c = patch.shape
        display[0:h, int(mon_w/2 - w/2):int(mon_w/2 + w/2), :] = 1.0 * patch / 255.0
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(display, '.', (x_pixel_gt, y_pixel_gt), font, 0.5, (0,0,0), 10, cv2.LINE_AA)
        cv2.putText(display, '.', (int(x_pixel_hat), int(y_pixel_hat)), font, 0.5, (0, 0, 255), 10, cv2.LINE_AA)
        # display = cv2.resize(display, None, fx=0.9, fy=0.9)
        cv2.namedWindow("por", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("por", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('por', display)

        # also show the face:
        cv2.rectangle(img, (int(face_location[0]), int(face_location[1])),
                      (int(face_location[2]), int(face_location[3])), (255, 0, 0), 2)
        landmarks_detector.plot_markers(img, pts)
        head_pose_estimator.drawPose(img, rvec, tvec, cam_calib['mtx'], np.zeros((1, 4)))
        cv2.imshow('image', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
