#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import sys
import cv2
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

sys.path.append("ext/HRNet-Facial-Landmark-Detection")
from lib.config import config
import lib.models as models
from lib.datasets import get_dataset
from lib.core import evaluation
from lib.utils import transforms

from face import face

class landmarks:

    def __init__(self, config=config):

        config.defrost()
        config.merge_from_file("ext/HRNet-Facial-Landmark-Detection/experiments/wflw/face_alignment_wflw_hrnet_w18.yaml")
        config.freeze()

        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.determinstic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED

        config.defrost()
        config.MODEL.INIT_WEIGHTS = False
        config.freeze()

        self.model = models.get_face_alignment_net(config)
        state_dict = torch.load("ext/HRNet-Facial-Landmark-Detection/hrnetv2_pretrained/HR18-WFLW.pth")
        self.model.load_state_dict(state_dict, strict=False)

        gpus = list(config.GPUS)
        self.model = nn.DataParallel(self.model, device_ids=gpus).cuda()

    def map_to_300vw(self):

        DLIB_68_PTS_MODEL_IDX = {
            "jaw": list(range(0, 17)),
            "left_eyebrow": list(range(17, 22)),
            "right_eyebrow": list(range(22, 27)),
            "nose": list(range(27, 36)),
            "left_eye": list(range(36, 42)),
            "right_eye": list(range(42, 48)),
            "left_eye_poly": list(range(36, 42)),
            "right_eye_poly": list(range(42, 48)),
            "mouth": list(range(48, 68)),
            "eyes": list(range(36, 42)) + list(range(42, 48)),
            "eyebrows": list(range(17, 22)) + list(range(22, 27)),
            "eyes_and_eyebrows": list(range(17, 22)) + list(range(22, 27)) + list(range(36, 42)) + list(range(42, 48)),
        }

        WFLW_98_PTS_MODEL_IDX = {
            "jaw": list(range(0, 33)),
            "left_eyebrow": list(range(33, 42)),
            "right_eyebrow": list(range(42, 51)),
            "nose": list(range(51, 60)),
            "left_eye": list(range(60, 68)) + [96],
            "right_eye": list(range(68, 76)) + [97],
            "left_eye_poly": list(range(60, 68)),
            "right_eye_poly": list(range(68, 76)),
            "mouth": list(range(76, 96)),
            "eyes": list(range(60, 68)) + [96] + list(range(68, 76)) + [97],
            "eyebrows": list(range(33, 42)) + list(range(42, 51)),
            "eyes_and_eyebrows": list(range(33, 42)) + list(range(42, 51)) + list(range(60, 68)) + [96] + list(
                range(68, 76)) + [97],
        }

        DLIB_68_TO_WFLW_98_IDX_MAPPING = OrderedDict()
        DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(0, 17), range(0, 34, 2))))  # jaw | 17 pts
        DLIB_68_TO_WFLW_98_IDX_MAPPING.update(
            dict(zip(range(17, 22), range(33, 38))))  # left upper eyebrow points | 5 pts
        DLIB_68_TO_WFLW_98_IDX_MAPPING.update(
            dict(zip(range(22, 27), range(42, 47))))  # right upper eyebrow points | 5 pts
        DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(27, 36), range(51, 60))))  # nose points | 9 pts
        DLIB_68_TO_WFLW_98_IDX_MAPPING.update({36: 60})  # left eye points | 6 pts
        DLIB_68_TO_WFLW_98_IDX_MAPPING.update({37: 61})
        DLIB_68_TO_WFLW_98_IDX_MAPPING.update({38: 63})
        DLIB_68_TO_WFLW_98_IDX_MAPPING.update({39: 64})
        DLIB_68_TO_WFLW_98_IDX_MAPPING.update({40: 65})
        DLIB_68_TO_WFLW_98_IDX_MAPPING.update({41: 67})
        DLIB_68_TO_WFLW_98_IDX_MAPPING.update({42: 68})  # right eye | 6 pts
        DLIB_68_TO_WFLW_98_IDX_MAPPING.update({43: 69})
        DLIB_68_TO_WFLW_98_IDX_MAPPING.update({44: 71})
        DLIB_68_TO_WFLW_98_IDX_MAPPING.update({45: 72})
        DLIB_68_TO_WFLW_98_IDX_MAPPING.update({46: 73})
        DLIB_68_TO_WFLW_98_IDX_MAPPING.update({47: 75})
        DLIB_68_TO_WFLW_98_IDX_MAPPING.update(dict(zip(range(48, 68), range(76, 96))))  # mouth points | 20 pts

        WFLW_98_TO_DLIB_68_IDX_MAPPING = {k: v for k, v in DLIB_68_TO_WFLW_98_IDX_MAPPING.items()}

        return list(WFLW_98_TO_DLIB_68_IDX_MAPPING.values())

    def detect(self, face_location, frame):

        x_min = face_location[0]
        y_min = face_location[1]
        x_max = face_location[2]
        y_max = face_location[3]

        w = x_max - x_min
        h = y_max - y_min
        scale = max(w, h) / 200
        scale *= 1.25

        center_w = (x_min + x_max) / 2
        center_h = (y_min + y_max) / 2
        center = torch.Tensor([center_w, center_h])

        frame_rgb = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.float32)
        img = transforms.crop(frame_rgb, center, scale, [256, 256], rot=0)

        img = img.astype(np.float32)
        img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img / 255.0 - img_mean) / img_std
        img = img.transpose([2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = torch.Tensor(img)

        self.model.eval()
        output = self.model(img)
        score_map = output.data.cpu()
        center = np.expand_dims(np.array(center, dtype=np.float32), axis=0)
        scale = np.expand_dims(np.array(scale, dtype=np.float32), axis=0)
        preds = evaluation.decode_preds(score_map, center, scale, [64, 64])
        preds = np.squeeze(preds.numpy(), axis=0)

        # get the 68 300 VW points:
        idx_300vw = self.map_to_300vw()
        preds = preds[idx_300vw, :]

        return preds

    def plot_markers(self, img, markers, color=(0, 0, 255), radius=3, drawline=False):
        # plot all 68 pts on the face image
        N = markers.shape[0]
        # if N >= 68:
        #     last_point = 68
        for i in range(0, N):
            x = markers[i, 0]
            y = markers[i, 1]
            # cv2.circle(img, (x, y), radius, color)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(i), (x, y), font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

        if drawline:
            def draw_line(start, end):
                for i in range(start, end):
                    x1 = markers[i, 0]
                    y1 = markers[i, 1]
                    x2 = markers[i + 1, 0]
                    y2 = markers[i + 1, 1]
                    cv2.line(img, (x1, y1), (x2, y2), color)

            draw_line(0, 16)
            draw_line(17, 21)
            draw_line(22, 26)
            draw_line(27, 35)
            draw_line(36, 41)
            draw_line(42, 47)
            draw_line(48, 67)

        return img


# landmarks_detector = landmarks()
#
# img = cv2.imread('test2.jpg')
# #img = cv2.resize(img, None, fx=0.5, fy=0.5)
#
# # detect the largest face:
# face_location = face.detect(img, use_max='SIZE')
#
# # detect facial points
# pts = landmarks_detector.detect(face_location, img)
#
# # display
# cv2.rectangle(img, (int(face_location[0]), int(face_location[1])), (int(face_location[2]), int(face_location[3])), (255, 0, 0), 2)
# landmarks_detector.plot_markers(img, pts, drawline=True)
# cv2.imwrite('test_out.png',img)
# cv2.imshow('test', img)
cv2.waitKey(0)
