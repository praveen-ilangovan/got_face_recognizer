# -*- coding: utf-8 -*-

"""
GOT Face Recognizer

Testing src.cv_utils module
"""

# Python builtin imports
import os

# Project specific imports
import cv2 as cv
import numpy as np

# Local imports
from src import cv_utils
from src import constants as Key

#-----------------------------------------------------------------------------#
#
# Tests
#
#-----------------------------------------------------------------------------#
def test_read_image(read_images):
    for img in read_images:
        assert isinstance(img, np.ndarray)

def test_convert_to_grayscale(read_images):
    for img in read_images:
        gray = cv_utils.convert_to_grayscale(img)
        assert len(gray.shape) == 2

def test_resize_image_height(read_images):
    for img in read_images:
        resized = cv_utils.resize_image(img, height=100)
        assert resized.shape[0] == 100

def test_resize_image_width(read_images):
    for img in read_images:
        resized = cv_utils.resize_image(img, width=100)
        assert resized.shape[1] == 100

def test_get_faces(read_images):
    for img in read_images:
        resized = cv_utils.resize_image(img, height=Key.RESIZE_HEIGHT)
        faces = cv_utils.get_faces(resized)
        assert len(faces) == 1

def test_haar_cascade_file_exists():
    assert os.path.exists(os.path.join(cv.data.haarcascades + Key.HAAR_CASCADE_FILE)) == True

def test_get_haar_cascade_classifier():
    assert isinstance(cv_utils.get_haar_cascade_classifier(), cv.CascadeClassifier)