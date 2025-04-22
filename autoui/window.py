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

        # Get rid of text to avoid detecting lines in it
        # image = erase_text_from_image(image)

        # Convert to greyscale, increase contrast a bit, and apply adaptive
        # thresholding to make finding element edges easier
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # grey = cv2.GaussianBlur(grey, (1, 1), 0)
        image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        image = cv2.adaptiveThreshold(
            image, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            3, 2
        )

        # Find all vertical or horizontal lines in image
        lines = cv2.HoughLinesP(
            thresh, 1, np.pi/2,
            threshold=50,
            minLineLength=40,
            maxLineGap=5
        )

        # Draw for debugging
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), thickness=1)
        show(image)

        # Find rectangles from these lines
        
        
        return

        lines = []
        for y in range(int(min_length/2), image.shape[0], min_length-1):
            y_slice = image[y]
            for x in range(0, image.shape[1]-1):
                window = y_slice[x:x+2]
                if not (window[0] == window[1]).all():
                    # Found a potential line
                    top, bottom = y, y
                    x_slice = image[:,x:x+2]
                    while top > 0:
                        if not (x_slice[top, 1] == x_slice[y, 1]).all() or\
                            (x_slice[top, 0] == x_slice[top, 1]).all():
                            break
                        top -= 1
                    while bottom < image.shape[0]:
                        if not (x_slice[bottom, 1] == x_slice[y, 1]).all() or\
                            (x_slice[bottom, 0] == x_slice[bottom, 1]).all():
                            break
                        bottom += 1
                    if bottom - top >= min_length:
                        lines.append((x+1, top, x+1, bottom))

        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), thickness=2)
        cv2.imshow("test", image)
        cv2.waitKey(0)
        return


    @staticmethod
    def get_regions(image):
        """Returns the rectangles of potential windows or widgets in the image."""

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Optional: blur to reduce noise
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Binarize: try adaptive threshold or simple thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        show(thresh)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = []

        for contour in contours:
            # Approximate contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the polygon has 4 sides and is convex
            if len(approx) == 4 and cv2.isContourConvex(approx):
                # Optional: check if it looks like a rectangle (right angles)
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                area = cv2.contourArea(approx)
                rect_area = w * h
                fill_ratio = area / rect_area

                # Heuristic filters
                if fill_ratio > 0.8 and area > 100:  # tweak as needed
                    rectangles.append(approx)
                    cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

        # Show result
        cv2.imshow('Rectangles', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return

        lines = []
        for y, row in enumerate(image):
            tail = row[0]
            tail_x = None
            for x, pixel in enumerate(row):
                above_pixel = image[max(min(y-1, image.shape[0]), 0)]
                if not (pixel == tail).all() or (above_pixel == pixel).all():
                    if tail_x and x - tail_x > 100:
                        lines.append((tail_x, y, x, y))

                    tail = pixel
                    tail_x = x



        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
        cv2.imshow("test", image)
        cv2.waitKey(0)
        return


        # Get edges in image
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grey, 30, 30)

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