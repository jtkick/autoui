#!/usr/bin/env python3

import cv2
import numpy as np
import pathlib
import pytest

import autoui

DATA_DIR = pathlib.Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def basic_square():
    return cv2.imread('data/basic_square.png')

@pytest.fixture(scope="session")
def terminal_window():
    return cv2.imread('data/terminal_window.png')

@pytest.fixture(scope="session")
def window_corners():
    return cv2.imread('data/window_corners.png')

def test_get_lines_basic_square(basic_square):
    # Prepare
    expected_horizontal = [
        np.array([10, 10, 40, 10]),
        np.array([10, 40, 39, 40])
    ]
    expected_vertical = [
        np.array([10, 10, 10, 40]),
        np.array([40, 10, 40, 39])
    ]

    # Act
    h, v = autoui.Region.get_lines(basic_square, min_length=10)

    # Assert
    assert((h == expected_horizontal).all())
    assert((v == expected_vertical).all())

def test_get_regions_terminal_window(terminal_window):
    # Prepare
    expected_window = ((62, 55), (499, 348))

    # Act
    r = autoui.Region.get_regions(terminal_window)

    # Assert
    assert(r == [expected_window])