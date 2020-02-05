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

import os
import cv2
import eos
import numpy as np

class EosHeadPoseEstimator(object):

    def __init__(self):
        cwd = os.path.dirname(__file__)
        base_dir = cwd + '/ext/eos'

        model = eos.morphablemodel.load_model(base_dir + '/share/sfm_shape_3448.bin')
        self.blendshapes = eos.morphablemodel.load_blendshapes(
            base_dir + '/share/expression_blendshapes_3448.bin')
        self.morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(
            model.get_shape_model(), self.blendshapes,
            eos.morphablemodel.PcaModel(),
            model.get_texture_coordinates(),
        )
        self.landmark_mapper = eos.core.LandmarkMapper(
            base_dir + '/share/ibug_to_sfm.txt')
        self.edge_topology = eos.morphablemodel.load_edge_topology(
            base_dir + '/share/sfm_3448_edge_topology.json')
        self.contour_landmarks = eos.fitting.ContourLandmarks.load(
            base_dir + '/share/ibug_to_sfm.txt')
        self.model_contour = eos.fitting.ModelContour.load(
            base_dir + '/share/sfm_model_contours.json')

    def fit_func(self, landmarks, image_size):
        image_w, image_h = image_size
        return eos.fitting.fit_shape_and_pose(
            self.morphablemodel_with_expressions, landmarks_to_eos(landmarks),
            self.landmark_mapper, image_w, image_h, self.edge_topology,
            self.contour_landmarks, self.model_contour,
        )


def landmarks_to_eos(landmarks):
    out = []
    for i, (x, y) in enumerate(landmarks[:68, :]):
        out.append(eos.core.Landmark(str(i + 1), [x, y]))
    return out


class PnPHeadPoseEstimator(object):
    ibug_ids_to_use = sorted([
        28, 29, 30, 31,  # nose ridge
        32, 33, 34, 35, 36,  # nose base
        37, 40,  # left-eye corners
        43, 46,  # right-eye corners
    ])

    def __init__(self):
        # Load and extract vertex positions for selected landmarks
        cwd = os.path.dirname(__file__)
        base_dir = cwd + '/ext/eos'
        self.model = eos.morphablemodel.load_model(
            base_dir + '/share/sfm_shape_3448.bin')
        self.shape_model = self.model.get_shape_model()
        self.landmarks_mapper = eos.core.LandmarkMapper(
            base_dir + '/share/ibug_to_sfm.txt')
        self.sfm_points_ibug_subset = np.array([
            self.shape_model.get_mean_at_point(
                int(self.landmarks_mapper.convert(str(d)))
            )
            for d in range(1, 69)
            if self.landmarks_mapper.convert(str(d)) is not None
        ])

        self.sfm_points_for_pnp = np.array([
            self.shape_model.get_mean_at_point(
                int(self.landmarks_mapper.convert(str(d)))
            )
            for d in self.ibug_ids_to_use
        ])

        # Rotate face around
        rotate_mat = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        self.sfm_points_ibug_subset = np.matmul(self.sfm_points_ibug_subset.reshape(-1, 3), rotate_mat)
        self.sfm_points_for_pnp = np.matmul(self.sfm_points_for_pnp.reshape(-1, 3), rotate_mat)

        # Center on mean point between eye corners
        between_eye_point = np.mean(self.sfm_points_for_pnp[-4:, :], axis=0)
        self.sfm_points_ibug_subset -= between_eye_point.reshape(1, 3)
        self.sfm_points_for_pnp -= between_eye_point.reshape(1, 3)

    def fit_func(self, landmarks, camera_parameters):
        landmarks = np.array([
            landmarks[i - 1, :]
            for i in self.ibug_ids_to_use
        ], dtype=np.float64)
        fx, fy, cx, cy = camera_parameters

        # Initial fit
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(self.sfm_points_for_pnp, landmarks,
                                                          camera_matrix, None, flags=cv2.SOLVEPNP_EPNP)

        # Second fit for higher accuracy
        success, rvec, tvec = cv2.solvePnP(self.sfm_points_for_pnp, landmarks, camera_matrix, None,
                                           rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

        return rvec, tvec

    def project_model(self, rvec, tvec, camera_parameters):
        fx, fy, cx, cy = camera_parameters
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        points, _ = cv2.projectPoints(self.sfm_points_ibug_subset, rvec, tvec, camera_matrix, None)
        return points


    def drawPose(self, img, r, t, cam, dist):

        modelAxes = np.array([
            np.array([0., -20., 0.]).reshape(1, 3),
            np.array([50., -20., 0.]).reshape(1, 3),
            np.array([0., -70., 0.]).reshape(1, 3),
            np.array([0., -20., -50.]).reshape(1, 3)
        ])

        projAxes, jac = cv2.projectPoints(modelAxes, r, t, cam, dist)

        cv2.line(img, (int(projAxes[0, 0, 0]), int(projAxes[0, 0, 1])),
                 (int(projAxes[1, 0, 0]), int(projAxes[1, 0, 1])),
                 (0, 255, 255), 2)
        cv2.line(img, (int(projAxes[0, 0, 0]), int(projAxes[0, 0, 1])),
                 (int(projAxes[2, 0, 0]), int(projAxes[2, 0, 1])),
                 (255, 0, 255), 2)
        cv2.line(img, (int(projAxes[0, 0, 0]), int(projAxes[0, 0, 1])),
                 (int(projAxes[3, 0, 0]), int(projAxes[3, 0, 1])),
                 (255, 255, 0), 2)
