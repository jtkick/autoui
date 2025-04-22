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

image = autoui.ui.screenshot()
print("got screenshot")
autoui.Region.get_lines(image)
print("got lines")
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