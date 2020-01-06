# pip install pgi

# seems to work from terminal outside of pycharm
import gi.repository
gi.require_version('Gdk', '3.0')
from gi.repository import Gdk
import numpy as np

class monitor:

    def __init__(self):
        display = Gdk.Display.get_default()
        screen = display.get_default_screen()
        default_screen = screen.get_default()
        num = default_screen.get_number()

        self.h_mm = default_screen.get_monitor_height_mm(num)
        self.w_mm = default_screen.get_monitor_width_mm(num)

        self.h_pixels = default_screen.get_height()
        self.w_pixels = default_screen.get_width()

    def monitor_to_camera(self, x_pixel, y_pixel):

        # assumes in-build laptop camera, located centered and 10 mm above display
        x_cam_mm = ((int(self.w_pixels/2) - x_pixel)/self.w_pixels) * self.w_mm
        y_cam_mm = 10.0 + (y_pixel/self.h_pixels) * self.h_mm
        z_cam_mm = 0.0

        return x_cam_mm, y_cam_mm, z_cam_mm

    def camera_to_monitor(self, x_cam_mm, y_cam_mm):

        x_mon_pixel = np.ceil(int(self.w_pixels/2) - x_cam_mm * self.w_pixels / self.w_mm)
        y_mon_pixel = np.ceil((y_cam_mm - 10.0) * self.h_pixels / self.h_mm)

        return x_mon_pixel, y_mon_pixel
