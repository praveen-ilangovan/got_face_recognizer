import pytest
from src import cv_utils

TEST_FILES = (
    "resources\\training\\Jon_Snow\\5.jpg",
    # "resources\\training\\Jon_Snow\\5.jpg",
    # "resources\\training\\Jon_Snow\\7.jpg",
    # "resources\\training\\Jon_Snow\\11.jpg",
    # "resources\\training\\Jon_Snow\\15.jpg"
)

@pytest.fixture(scope="session")
def read_images():
    return [cv_utils.read_image(f) for f in TEST_FILES]