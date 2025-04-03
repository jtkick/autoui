import cv2
import datetime
import easyocr
import math
from pynput import mouse
import numpy as np
from playsound import playsound
import pyautogui
# import pyglm
import random
from skimage.metrics import structural_similarity as ssim
import threading
import time

from . import tools
from . import ui


import numpy as np
import matplotlib.pyplot as plt


# Position = pyglm.glm.uvec2

class Region:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.w = 1
        self.h = 1

    @staticmethod
    def get_regions(image):
        """Returns the rectangles of potential windows or widgets in the image."""
        # Get edges in image
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grey, 30, 80)

        cv2.imshow("test", edges)
        cv2.waitKey(0)

        # reader = easyocr.Reader(["en"])
        # results = reader.readtext(image)
        # for (bbox, text, prob) in results:
        #     # print(bbox)
        #     #print(f"Detected text: {text} (Confidence: {prob:.4f})")
        #     cv2.rectangle(edges, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (0, 0, 0), thickness=-1)
        # # cv2.imshow("test", edges)
        # # cv2.waitKey(0)
        # # return


        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=500, minLineLength=100, maxLineGap=50)

        mask = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), thickness=2)
        cv2.imshow("test", image)
        cv2.waitKey(0)
        return

        # Find rectangles in image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:  # A rectangle should have 4 corners
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 2.0:  # Avoid extreme aspect ratios
                    rectangles.append((x, y, w, h))
                    cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)  # Draw rectangle in green

        # Draw rectangles for testing
        # for rect in rectangles:
        #     cv2.rectangle(image, rect[0:2], (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 255), 2)


        # # Create a blank mask
        # mask = np.zeros_like(image)

        # if lines is not None:
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]
        #         cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), thickness=2)


        cv2.imshow("test", image)
        cv2.waitKey(0)

class Window(Region):

    def __init__(self, open=None, timeout=5.0):
        def default():
            pass
        self.open = open if open else default
        self.num = 0
        self._parent_window = None
        self._timeout = timeout

    def __enter__(self):

        # Get picture of screen before window opens
        before = ui.screenshot()

        # Do whatever the user specified to open the window
        self.open()

        # Flags for tracking when to stop watching screen for changes
        changing = False
        done_changing = False
        changed = before

        # Now we start looking for the window to appear
        start_time = time.time()
        while time.time() - start_time < self._timeout:

            # Get difference between screen now and when we started looking
            after = ui.screenshot()
            time.sleep(0.1)

            # Wait for the screen to change in some way
            if not changing:
                # print("waiting for changes")
                diff = cv2.absdiff(before, after)
                changing = np.sum(diff) / diff.size > 0.05
                continue

            # Wait for screen to stop changing
            if not done_changing:
                # print("waiting for changes to stop")
                diff = cv2.absdiff(changed, after)
                done_changing = np.sum(diff) / diff.size < 0.05
                changed = after
                continue

            after = np.maximum(after.astype(np.int16) - before, 0).astype(np.uint8)

            # Convert to grayscale
            gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter rectangular contours
            rectangles = []
            for cnt in contours:
                approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                if len(approx) == 4:  # A rectangle should have 4 corners
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 2.0:  # Avoid extreme aspect ratios
                        rectangles.append((x, y, w, h))

            rectangles = [r for r in rectangles if r[2] >= 200 and r[3] >= 200]

            if len(rectangles) == 1:
                (self.x, self.y, self.w, self.h) = rectangles[0]
                break

        else:
            raise RuntimeError("Window now found.")

        return self

    def __exit__(self, exc_type, exc_value, tb):
        pass

    # Takes a coordinate and wraps it around to the other side if negative
    def _rel_to_abs_pos(self, pos):
        return (self.x + (pos[0] % self.w), self.y + (pos[1] % self.h))

    def move_mouse(self, pos, *args, **kwargs):
        ui.move_mouse(self._rel_to_abs_pos(pos), *args, **kwargs)

    def screenshot(self, top_left=(0, 0), bottom_right=(-1, -1)):
        ui.screenshot(tl=self._rel_to_abs_pos(top_left),
                      br=self._rel_to_abs_pos(bottom_right))

    def temp_text(self):
        image = self.screenshot()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        reader = easyocr.Reader(["en"])
        results = reader.readtext(image)
        for (bbox, text, prob) in results:
            print(f"Detected text: {text} (Confidence: {prob:.4f})")