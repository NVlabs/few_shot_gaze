# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello.
# --------------------------------------------------------

import sys
import cv2

sys.path.append("ext/mtcnn-pytorch/")
from src import detect_faces, show_bboxes
from PIL import Image

class face:

    def detect(frame, use_max='SIZE'):

        # detect face
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

        return face_location

