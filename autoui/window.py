import cv2
import datetime
import easyocr
import math
import pathlib
from pynput import mouse
import numpy as np
from playsound import playsound
import pyautogui
import pytesseract
# import pyglm
import random
from skimage.metrics import structural_similarity as ssim
import threading
import time
import ultralytics

from . import tools
from . import ui


import numpy as np
import matplotlib.pyplot as plt

def show(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)

# Position = pyglm.glm.uvec2


# def erase_text_from_image(image, lang='eng'):
#     # Run pytesseract to get bounding boxes of detected text
#     data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)

#     n_boxes = len(data['level'])
#     for i in range(n_boxes):
#         x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

#         if w == 0 or h == 0:
#             continue

#         # Define padding around text box for sampling
#         pad = 2
#         y1, y2 = max(0, y - pad), min(image.shape[0], y + h + pad)
#         x1, x2 = max(0, x - pad), min(image.shape[1], x + w + pad)

#         # Create a mask for the text region
#         text_mask = np.zeros(image.shape[:2], dtype=np.uint8)
#         cv2.rectangle(text_mask, (x, y), (x + w, y + h), 255, -1)

#         # Sample surrounding region (excluding text)
#         surrounding = image[y1:y2, x1:x2].copy()
#         surrounding_mask = text_mask[y1:y2, x1:x2] == 0
#         if np.count_nonzero(surrounding_mask) == 0:
#             continue

#         sampled_pixels = surrounding[surrounding_mask]
#         if len(sampled_pixels) == 0:
#             continue

#         # Use the median surrounding color
#         median_color = np.median(sampled_pixels, axis=0).astype(np.uint8).tolist()

#         # Fill the text region with that color
#         cv2.rectangle(output, (x, y), (x + w, y + h), median_color, -1)

#     return output


class Line:
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def contains(self, point):
        pass

class Region:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.w = 1
        self.h = 1

    @staticmethod
    def get_lines(image, min_length=10):

        # Convert to int16 explicitly before doing *any* diff
        img = image.astype(np.int16)
        h, w, _ = img.shape

        # Compute the difference between each pixel and its right & bottom neighbors
        diff_x = img[:, 1:, :] - img[:, :-1, :]
        diff_y = img[1:, :, :] - img[:-1, :, :]

        # Pad to original size (since diff loses a row/col)
        edge_map = np.zeros(img.shape, dtype=np.uint8)
        edge_map[:, 1:] |= np.abs(diff_x).astype(np.uint8)
        edge_map[1:, :] |= np.abs(diff_y).astype(np.uint8)

        # Convert to greyscale, increase contrast a bit, and apply adaptive
        # thresholding to make finding element edges easier
        grey = cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)
        grey = cv2.convertScaleAbs(grey, alpha=2.5, beta=-1)

        # Clear out likely shadows and noise
        EDGE_THRESHOLD = 10
        binary = np.where(grey > EDGE_THRESHOLD, 1, 0).astype(np.uint8)

        # Find all straight lines in image longer than the MIN_LINE_LENGTH
        MIN_LINE_LENGTH = min_length
        horizontal = []
        vertical = []
        for y in range(h):
            row = binary[y, :]
            changes = np.diff(np.concatenate(([0], row, [0])))
            starts = np.where(changes == 1)[0]
            ends   = np.where(changes == -1)[0]
            for s, e in zip(starts, ends):
                if e - s >= MIN_LINE_LENGTH:
                    horizontal.append((s, y, e - 1, y))  # row y, from s to e-1
        for x in range(w):
            col = binary[:, x]
            changes = np.diff(np.concatenate(([0], col, [0])))
            starts = np.where(changes == 1)[0]
            ends   = np.where(changes == -1)[0]
            for s, e in zip(starts, ends):
                if e - s >= MIN_LINE_LENGTH:
                    vertical.append((x, s, x, e - 1))  # column x, from s to e-1

        return horizontal, vertical
            


    @staticmethod
    def get_regions(image):
        """Returns the rectangles of potential windows or widgets in the image."""

        # Find lines in image
        horizontal, vertical = Region.get_lines(image, min_length = 10)

        # Find rectangles from these lines
        MAX_RADIUS = 20
        ATTACHMENT_DISTANCE = 5
        for top in horizontal:
            # 'top' is the starting line, i.e., the top of the rectangle
            # So the first step is to find the right side of the rectangle
            candidates = [
                line for line in vertical
                if -3 < line[1] - top[3] < ATTACHMENT_DISTANCE
            ]

class Window(Region):

    def __init__(self, open=None, timeout=5.0):
        def default():
            pass
        self.open = open if open else default
        self.num = 0
        self._parent_window = None
        self._timeout = timeout

        model_path = pathlib.Path(__file__).resolve().parent
        model_path = model_path / "assets" / "weights" / "windows.pt"
        self._model = ultralytics.YOLO(str(model_path))

    def __enter__(self):

        # Get picture of screen before window opens
        before = ui.screenshot()

        # self.open()
        time.sleep(2)
        before = ui.screenshot()


        results = self._model(before, conf=0.80)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = self._model.names[cls_id]
            if label == "window":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Draw the bounding box
                cv2.rectangle(before, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put class label and confidence
                cv2.putText(before, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Display the image
        cv2.imshow("Detections", before)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()

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

    def mouse_down(self, *args, **kwargs):
        ui.mouse_down(*args, **kwargs)

    def mouse_up(self, *args, **kwargs):
        ui.mouse_up(*args, **kwargs)

    def move_mouse(self, pos, *args, **kwargs):
        ui.move_mouse(self._rel_to_abs_pos(pos), *args, **kwargs)

    def resize(self, dimensions):
        self.move_mouse((-1, -1))
        self.mouse_down()
        ui.move_mouse((self.x + dimensions[0], self.y + dimensions[1]))
        self.mouse_up()

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