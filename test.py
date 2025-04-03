#!/usr/bin/env python3

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

# image = autoui.Screen().screenshot()
# autoui.Region.get_regions(image)
# exit()

with autoui.Process("gnome-calculator") as app:
    with autoui.Window(app.start()) as win:
        win.move_mouse((0, 0), careful=True)
        time.sleep(1)
        win.move_mouse((0, -1))
        time.sleep(1)
        win.move_mouse((-1, -1), careful=True)
        time.sleep(1)
        win.move_mouse((-1, 0))