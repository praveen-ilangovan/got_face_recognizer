import pytest
import numpy as np

from src import cv_utils, constants

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
        faces = cv_utils.get_faces(img)
        assert len(faces) == 1