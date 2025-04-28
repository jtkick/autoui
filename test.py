#!/usr/bin/env python3

import cv2
import time

import autoui

import numpy as np
import matplotlib.pyplot as plt

# image = autoui.ui.screenshot()
image = cv2.imread('./test_box.png')

# Test get lines
# h, v = autoui.Region.get_lines(image)
# print('h:', h)
# print('v:', v)
# print()
# autoui.Region.show_lines(image)

# Test get regions
autoui.Region.get_regions(image)

exit()

with autoui.Process("gnome-calculator") as app:
    with autoui.Window(app.start()) as win:
        win.move_mouse((0, 0))
        time.sleep(1)
        win.move_mouse((0, -1))
        time.sleep(1)
        win.move_mouse((-1, -1))
        time.sleep(1)
        win.move_mouse((-1, 0))

        win.resize((500, 500))
        time.sleep(5)