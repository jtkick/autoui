import datetime
import math
import mouse
import numpy as np
import playsound
import pyautogui
import pynput
import random
import threading
import time

from . import tools


class _UI:

    def __init__(self):
        
        # If this ever locks, autoui has been disengaged by the user, so wait
        # to do anything until it's re-engaged
        self.engaged = threading.Lock()

        # These are for 
        self.mouse_lock = threading.Lock()
        self.last_mouse_pos = pyautogui.position()
        self.last_mouse_move_time = datetime.datetime.now()
        self.mouse_running_diff = 0

        # Initialize module
        listener = pynput.mouse.Listener(on_move=self._on_mouse_move)
        listener.start()

    def key_down(self, key):
        pyautogui.keyDown(key)

    def key_up(self, key):
        pyautogui.keyUp(key)

    def mouse_down(self, *args, **kwargs):
        pyautogui.mouseDown(*args, **kwargs)

    def mouse_up(self, *args, **kwargs):
        pyautogui.mouseUp(*args, **kwargs)

    def move_mouse(self, pos, realistic=True, careful=False):
        """Moves the mouse to the given position on the screen."""
        # Just go straight there if we don't have to be realistic
        if not realistic:
            with self.mouse_lock:
                self._check_engagement()
                mouse.move(pos[0], pos[1], absolute=True)
                self.last_mouse_pos = pos

        def get_mouse_curve(start_pos, end_pos):
            control_points = [start_pos]
            # Get a control point in the direction of the end position
            diff_pos = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
            diff_dist = int(math.sqrt(diff_pos[0] ** 2 + diff_pos[1] ** 2))
            diff_avg = abs(diff_pos[0] + diff_pos[1]) / 2
            control_points.append((
                start_pos[0] + (diff_pos[0] * random.uniform(0, 1)),
                start_pos[1] + (diff_pos[1] * random.uniform(0, 1))
            ))
            # Add control point around the end point
            if not careful:
                control_points.append((
                    start_pos[0] + (diff_pos[0]/2) + (diff_avg * random.uniform(0, 1)),
                    start_pos[1] + (diff_pos[1]/2) + (diff_avg * random.uniform(0, 1))
                ))
            control_points.append(end_pos)
            control_points = np.array(control_points)

            # Generate the curve
            bias = 10.0 if not careful else 3.0
            curve = tools.bezier_curve(control_points, n_points=500, bias=bias)

            # plt.plot(curve[:, 0], curve[:, 1], 'b-', label="Bézier Curve")
            # plt.plot(control_points[:, 0], control_points[:, 1], 'ro--', label="Control Points")
            # plt.legend()
            # plt.xlabel("x")
            # plt.ylabel("y")
            # plt.title("Bézier Curve with More Points at Start")
            # plt.show()

            return curve

        # Get position and difference
        mouse_pos = pyautogui.position()
        total_dist = math.sqrt((pos[0]-mouse_pos[0])**2 + (pos[1]-mouse_pos[1])**2)

        # For realism, miss and readjust every once in awhile
        points = []
        if random.randint(0, 2) == 0 and not careful:
            points.append((pos[0] + random.randint(-100, 100), pos[1] + random.randint(-100, 100)))
        points.append(pos)

        # Move mouse to each point
        speed = 0.0003 if careful == False else 0.002
        for pos in points:
            curve = get_mouse_curve(pyautogui.position(), pos)
            for x, y in curve:
                with self.mouse_lock:
                    self._check_engagement()
                    mouse.move(x,y,absolute=True, duration=speed)
                    self.last_mouse_pos = pyautogui.Point(int(x), int(y))
            time.sleep(random.uniform(0.001, 0.1))

    def screenshot(self, tl=None, br=None):
        """Takes a screenshot of the specified rectangle and returns an image in
        CV2 format."""
        if tl and br:
            region = (tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])
            img = pyautogui.screenshot(region=region)
        else:
            img = pyautogui.screenshot()
        # Convert to cv2 image
        img = np.array(img.convert('RGB'))
        img = img[:, :, ::-1].copy()
        return img

    def _on_mouse_move(self):
        """Watches mouse movements. If the user is trying to move it, disable
        the screen class until the user re-enables it or exits the program."""
        with self.mouse_lock:
            pos = pyautogui.position()
            # If we moved it, just ignore it
            # print(type(pyautogui.position()))
            # print(type(_last_mouse_pos))
            if pyautogui.position() == self.last_mouse_pos:
                return

            # Keep track of how many pixels we've moved per second
            now = datetime.datetime.now()
            diff = (now - self.last_mouse_move_time).total_seconds() * 1000
            self.mouse_running_diff = max(self.mouse_running_diff - diff, 0)

            # If it wasn't us who moved it, calculate difference in position        
            self.mouse_running_diff += math.sqrt(
                (pos.x - self.last_mouse_pos.x) ** 2 +
                (pos.y - self.last_mouse_pos.y) ** 2
            )

            # If the running diff ever goes above the limit, that means the user is
            # trying to move it, and thus, autoui should disengage
            if self.mouse_running_diff > 10:
                # playsound.playsound('/home/jared/Development/autoui/autoui/assets/disengage.mp3')
                self.mouse_running_diff = 0
                self.engaged.acquire()

        move_mouse(self.last_mouse_pos, realistic=False)

    def _check_engagement(self):

        if not self.engaged.locked():
            return
        print("AutoUI has been disengaged. Press 'Enter' to re-engage.")
        # TODO: LOCK THREAD PROPERLY
        time.sleep(9999999)
        self.engaged.release()
        print("Resuming control...")

_ui = _UI()

key_down = _ui.key_down
key_up = _ui.key_up
mouse_down = _ui.mouse_down
mouse_up = _ui.mouse_up
move_mouse = _ui.move_mouse
screenshot = _ui.screenshot