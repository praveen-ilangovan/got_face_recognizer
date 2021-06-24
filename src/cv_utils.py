from typing import Optional
import cv2 as cv #type: ignore
import numpy as np

from . import utils
from . import constants

#-----------------------------------------------------------------------------#
#
# Read the image
#
#-----------------------------------------------------------------------------#
def read_image(imgpath: str) -> Optional[np.ndarray]:
    """ Reads the incoming image using cv.imread.

    Args:
        imgpath str: Path to an image file

    rtype:
        numpy.ndarray

    Returns:
        A numpy array
    """
    if not utils.file_exists(imgpath):
        print("File not exist: {0}".format(imgpath))
        return None
    return cv.imread(imgpath)

#-----------------------------------------------------------------------------#
#
# Image processing
#
#-----------------------------------------------------------------------------#
def convert_to_grayscale(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """ Convert the image's colour space to grayscale
    """
    if img is None:
        return
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#-----------------------------------------------------------------------------#
#
# Display the image
#
#-----------------------------------------------------------------------------#
def show_image(img: Optional[np.ndarray], window_name: str= constants.NAME, wait: int= 0) -> None:
    """ Displays the image using cv.imshow method

    Args:
        img np.ndarray: An opencv image stored as numpy array
        window_name str: Name of the window
        wait int: Number of milliseconds to wait before the window closes.
            Defaults to 0 and this makes sure the window shows until the
            user closes it.
    """
    if img is None:
        return

    cv.imshow(window_name, img)
    cv.waitKey(wait)

def display_image(imgpath: str, window_name: str= constants.NAME, wait: int= 0) -> None:
    """ Reads and shows the image using opencv module

    Args:
        imgpath str: Path to an image file
        window_name str: Name of the window
        wait int: Number of milliseconds to wait before the window closes.
            Defaults to 0 and this makes sure the window shows until the
            user closes it.
    """
    show_image(read_image(imgpath))