import cv2 as cv #type: ignore
import numpy as np

from . import utils

def read_image(imgpath: str) -> np.ndarray:
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

def show_image(img: np.ndarray, window_name: str= "GOT Face Recognizer", wait: int= 0) -> None:
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