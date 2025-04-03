"""Description."""

import pathlib
import re
import subprocess

from .window import *

class Process:

    def __init__(self, command: str):
        self._command = command
        self._process = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.stop()

    def start(self):
        if isinstance(self._command, str):
            command = self._command.split()
        self._process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def stop(self):
        self._process.terminate()
        self._process.wait()

class Application:

    def __init__(self, command: str):
        self._process = Process(command)

    def __enter__(self):
        with Window(self._process.start()) as w:
            pass

        return self

    def __exit__(self, exc_type, exc_value, tb):
        self._process.stop()

    def window(self):
        """Returns the window associated with the application, or the first
        window if there are multiple."""
        raise NotImplementedError()

    def windows(self):
        """Returns a list of all windows associated with the application."""
        raise NotImplementedError()

    @staticmethod
    def find(name: str = "", timeout: float = 5.0):
        """Looks for an open window with the given name."""
        # Get window information with 'xwininfo' command if on Linux
        window_info = None
        try:
            window_info, err = subprocess.Popen(
                f"xwininfo -name {name}",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True
            ).communicate()
        except:
            pass

        if not window_info:
            raise RuntimeError('Window not found')

        # Parse window stats for location and size
        window_info = window_info.decode('utf-8')
        match = re.search(r'(?<=Absolute upper-left X:  )\d*', window_info)
        self.x = int(match.group())
        match = re.search(r'(?<=Absolute upper-left Y:  )\d*', window_info)
        self.y = int(match.group())
        match = re.search(r'(?<=Width: )\d*', window_info)
        self.width = int(match.group())
        match = re.search(r'(?<=Height: )\d*', window_info)
        self.height = int(match.group())