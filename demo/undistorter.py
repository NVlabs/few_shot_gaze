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


import cv2 as cv
import numpy as np

class Undistorter(object):

    def __init__(self, camera_matrix, distortion_coefficients, output_size=None,
                 new_camera_matrix=None):
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.output_size = output_size
        self.new_camera_matrix = new_camera_matrix
        self.undistort_maps = None

    def apply(self, image_original):
        if self.undistort_maps is None:
            h, w, _ = image_original.shape
            self.undistort_maps = cv.initUndistortRectifyMap(
                self.camera_matrix, self.distortion_coefficients, np.eye(3), self.new_camera_matrix,
                (w, h) if self.output_size is None else self.output_size, cv.CV_32FC1)
        return cv.remap(image_original, self.undistort_maps[0], self.undistort_maps[1],
                        cv.INTER_LINEAR)
