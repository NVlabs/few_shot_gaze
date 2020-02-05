#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import sys
import cv2

sys.path.append("ext/mtcnn-pytorch/")
from src import detect_faces, show_bboxes
from PIL import Image

class face:

    def detect(frame, scale = 1.0, use_max='SIZE'):

        # detect face
        frame_small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(frame_rgb)
        bounding_boxes, landmarks = detect_faces(pil_im, min_face_size=30.0)
        dets = [x[:4] for x in bounding_boxes]
        scores = [x[4] for x in bounding_boxes]

        face_location = []
        if len(dets) > 0:
            max = 0
            max_id = -1
            for i, d in enumerate(dets):
                if use_max == 'SCORE':
                    property = scores[i]
                elif use_max == 'SIZE':
                    property = abs(dets[i][2] - dets[i][0]) * abs(dets[i][3] - dets[i][1])
                if max < property:
                    max = property
                    max_id = i
            if use_max == 'SCORE':
                if max > -0.5:
                    face_location = dets[max_id]
            else:
                face_location = dets[max_id]
            face_location = face_location * (1/scale)

        return face_location

