#!/usr/bin/env python3

import cv2
import time

import autoui

import numpy as np
import matplotlib.pyplot as plt

# with autoui.Screen() as s:
#     s.move_mouse((10, 10))
#     time.sleep(1)
#     s.move_mouse((-10, 10))
#     time.sleep(1)
#     s.move_mouse((10, -10))
#     time.sleep(1)
#     s.move_mouse((-10, -10))
    
# exit()

image = autoui.ui.screenshot()
print("got screenshot")

# Test get lines
h, v = autoui.Region.get_lines(image)
for line in h:
    x1, y1, x2, y2 = line
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)
for line in v:
    x1, y1, x2, y2 = line
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
cv2.imshow("lines", image)
cv2.waitKey(0)

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