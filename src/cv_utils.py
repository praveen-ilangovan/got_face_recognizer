import os
from typing import List, Optional

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

    Args:
        img numpy.ndarray: Image in an numpy array format

    rtype:
        numpy.ndarray

    Returns:
        A numpy array
    """
    if img is None:
        return None
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def resize_image(img: Optional[np.ndarray], width: int= -1,
                 height: int = -1, inter = cv.INTER_AREA) -> Optional[np.ndarray]:
    """ Resize the image by keeping its aspect ratio

    Args:
        img numpy.ndarray: Image in an numpy array format
        width int: Width to resize to
        height int: Height to resize to
        inter: opencv interpolation to use.

    rtype:
        numpy.ndarray

    Returns:
        A numpy array
    """
    if img is None:
        return None

    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = img.shape[:2]

    if width == -1 and height == -1:
        return img

    if width == -1:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(img, dim, interpolation = inter)

    # return the resized image
    return resized

def get_faces(img: Optional[np.ndarray], classifier: str= constants.HAAR_CASCADE) -> List:
    """ Detect faces in the image using the specified classifier

    Args:
        img numpy.ndarray: Image in an numpy array format
        classifier str: Classifier to use

    rtype:
        List

    Returns:
        A list of rectangle coordinates: [[x,y,w,h]]
    """
    # convert it to gray
    gray = convert_to_grayscale(img)

    # Detect faces
    cascade_classifier = get_classifer(classifier)
    if cascade_classifier is None:
        return []
    return cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

#-----------------------------------------------------------------------------#
#
# Classifiers
#
#-----------------------------------------------------------------------------#
def get_classifer(classifier: str) -> Optional[cv.CascadeClassifier]:
    """ Return the cascade classifier
    """
    if classifier == constants.HAAR_CASCADE:
        return get_haar_cascade_classifier()
    return None

def get_haar_cascade_classifier() -> cv.CascadeClassifier:
    """ Return haar cascade classifier

    rtype:
        cv.CascadeClassifier
    """
    cascade_file = os.path.join(constants.CLASSIFIERS_DIR, "haar_face.xml")
    return cv.CascadeClassifier(cascade_file)

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
    img = read_image(imgpath)
    img = resize_image(img, height=constants.RESIZE_HEIGHT)
    rects = get_faces(img)

    for x,y,w,h in rects:
        cv.rectangle(img, (x,y), (x+w, y+h), (0,128,0), thickness=2)

    show_image(img)