# -*- coding: utf-8 -*-

"""
GOT Face Recognizer

Variables, functions and fixtures to share across the test modules.
"""

# Project specific imports
import pytest

# Local imports
from src import cv_utils

#-----------------------------------------------------------------------------#
#
# Test data
#
#-----------------------------------------------------------------------------#
TEST_FILES = (
    "resources\\training\\Jon_Snow\\1.jpg",
    "resources\\training\\Jon_Snow\\5.jpg",
    "resources\\training\\Jon_Snow\\7.jpg",
    "resources\\training\\Jon_Snow\\11.jpg",
    "resources\\training\\Jon_Snow\\15.jpg"
)

INVALID_FILES = (
    "resources\\training\\JonSnow\\1.jpg",
    "resources\\training\\JonSnow\\5.jpg",
)

#-----------------------------------------------------------------------------#
#
# Fixtures
#
#-----------------------------------------------------------------------------#
@pytest.fixture(scope="session")
def read_images():
    """
    Reads the test images and returns a list of images in numpy array type

    Uses new := operator.. this is to store temporary variables while creating
    a list.
    """
    return [img for f in TEST_FILES if (img := cv_utils.read_image(f)) is not None]