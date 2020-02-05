"""
Copyright 2019 ETH Zurich, Seonwook Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

#!/usr/bin/env python3


import cv2
import numpy as np

from head import PnPHeadPoseEstimator
head_pose_estimator = PnPHeadPoseEstimator()

def common_pre(entry, head_pose):

    rvec, tvec = head_pose
    if rvec is None or tvec is None:
        raise ValueError('rvec or tvec is None')

    # Calculate rotation matrix and euler angles
    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)
    rotate_mat, _ = cv2.Rodrigues(rvec)

    # Reconstruct frame
    full_frame = cv2.cvtColor(entry['full_frame'], cv2.COLOR_BGR2RGB)

    # Form camera matrix
    fx, fy, cx, cy = entry['camera_parameters']
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                             dtype=np.float64)

    # Get camera parameters
    normalized_parameters = {
        'focal_length': 1300,
        'distance': 600,
        'size': (256, 64),
    }
    n_f = normalized_parameters['focal_length']
    n_d = normalized_parameters['distance']
    ow, oh = normalized_parameters['size']
    norm_camera_matrix = np.array([[n_f, 0, 0.5*ow], [0, n_f, 0.5*oh], [0, 0, 1]],
                                  dtype=np.float64)

    # Compute gaze-origin (g_o)
    landmarks_3d = np.matmul(rotate_mat, head_pose_estimator.sfm_points_for_pnp.T).T + tvec.T
    g_o = np.mean(landmarks_3d[10:12, :], axis=0)
    g_o = g_o.reshape(3, 1)

    g_t = g = None
    if entry['3d_gaze_target'] is not None:
        g_t = entry['3d_gaze_target'].reshape(3, 1)
        g = g_t - g_o
        g /= np.linalg.norm(g)

    return [full_frame, rvec, tvec, rotate_mat, camera_matrix, n_f, n_d,
            norm_camera_matrix, ow, oh, landmarks_3d, g_o, g_t, g]


def normalize(entry, head_pose):
    [full_frame, rvec, tvec, rotate_mat, camera_matrix, n_f, n_d, norm_camera_matrix,
     ow, oh, landmarks_3d, g_o, g_t, g] = common_pre(entry, head_pose)

    # Code below is an adaptation of code by Xucong Zhang
    # https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/revisiting-data-normalization-for-appearance-based-gaze-estimation/

    distance = np.linalg.norm(g_o)
    z_scale = n_d / distance
    S = np.eye(3, dtype=np.float64)
    S[2, 2] = z_scale

    hRx = rotate_mat[:, 0]
    forward = (g_o / np.linalg.norm(g_o)).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T # rotation matrix R

    W = np.dot(np.dot(norm_camera_matrix, S),
               np.dot(R, np.linalg.inv(camera_matrix))) # transformation matrix
    patch = cv2.warpPerspective(full_frame, W, (ow, oh)) # image normalization

    R = np.asmatrix(R)

    # Correct head pose
    head_mat = R * rotate_mat
    n_h = np.array([np.arcsin(head_mat[1, 2]), np.arctan2(head_mat[0, 2], head_mat[2, 2])])

    # Correct head pose
    n_g = []
    if g is not None:
        # Correct gaze
        n_g = correctGaze(R, g)

    return patch, n_h, n_g, np.transpose(R), g_o, g_t


def correctGaze(R, g):
    n_g = R * g
    n_g /= np.linalg.norm(n_g)
    n_g = vector_to_pitchyaw(-n_g.T).flatten()
    return n_g


def vector_to_pitchyaw(vectors):
    """Convert given gaze vectors to yaw (theta) and pitch (phi) angles."""
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


def draw_gaze(image_in, eye_pos, pitchyaw, length=40.0, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out