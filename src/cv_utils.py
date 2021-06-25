# -*- coding: utf-8 -*-

"""
GOT Face Recognizer

Module that has all the open cv functionalities used in this
project.
"""

# Python built-in imports
import os
from typing import List, Optional

# Project specific imports
# "type:ignore" -> cv2 doesn't have type annotations stubs.
# setting type to ignore let mypy to ignore this module.
import cv2 as cv #type: ignore
import numpy as np

# Local imports
from . import utils
from . import constants as Key

#-----------------------------------------------------------------------------#
#
# Read the image
#
#-----------------------------------------------------------------------------#
def read_image(imgpath: str) -> Optional[np.ndarray]:
    """ Reads the incoming image using cv.imread

    Args:
        imgpath str: Path to an image file

    rtype:
        numpy.ndarray|None

    Returns:
        A numpy array or None
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
def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """ Convert the image's colour space to grayscale

    Args:
        img numpy.ndarray: Image in an numpy array format

    rtype:
        numpy.ndarray

    Returns:
        A numpy array
    """
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def resize_image(img: np.ndarray, width: int= -1,
                 height: int = -1, inter = cv.INTER_AREA) -> np.ndarray:
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

#-----------------------------------------------------------------------------#
#
# Classifiers
#
#-----------------------------------------------------------------------------#
def get_classifer(classifier: str) -> Optional[cv.CascadeClassifier]:
    """ Return the cascade classifier

    Args:
        classifier str: Name of the classifier to use

    rtype:
        cv.CascadeClassifier | None

    Returns:
        Requested classifier
    """
    if classifier == Key.HAAR_CASCADE:
        return get_haar_cascade_classifier()
    return None

def get_haar_cascade_classifier() -> cv.CascadeClassifier:
    """ Return haar cascade classifier

    rtype:
        cv.CascadeClassifier
    """
    return cv.CascadeClassifier(cv.data.haarcascades + Key.HAAR_CASCADE_FILE)

#-----------------------------------------------------------------------------#
#
# Face Recognizion
#
#-----------------------------------------------------------------------------#
def prepare_image(imgpath: str,
                  resize_width: int=-1,
                  resize_height: int=Key.RESIZE_HEIGHT) -> Optional[np.ndarray]:
    """ Prepare the image to get faces from it.
    Reads the image from the path and resizes it

    Args:
        imgpath str: Path to an image file
        width int: Width to resize to
        height int: Height to resize to

    rtype:
        numpy.ndarray | None

    Returns:
        A numpy array
    """
    img = read_image(imgpath)
    if img is None:
        return None

    return resize_image(img, width=resize_width, height=resize_height)

def get_faces(img: np.ndarray, classifier: str= Key.HAAR_CASCADE) -> List:
    """ Detect faces in the image using the specified classifier and return
    a list of face coordinates (rectangle (x,y,w,h)) and cropped gray scale
    images of the face

    Args:
        img numpy.ndarray: Image in an numpy array format
        classifier str: Classifier to use

    rtype:
        List

    Returns:
        A list of rectangle coordinates and
        cropped grayscale image of face: [[x,y,w,h], grayscale_img]
    """
    # convert it to gray
    gray = convert_to_grayscale(img)

    # Detect faces
    cascade_classifier = get_classifer(classifier)
    if cascade_classifier is None:
        return []
    rects = cascade_classifier.detectMultiScale(gray,
                scaleFactor=1.1, minNeighbors=4)

    # For each rect, crop the face
    return [((x,y,w,h), gray[y:y+h, x:x+w]) for x,y,w,h in rects]

def train_model(features: np.ndarray, labels: np.ndarray,
                savepath: Optional[str]=None) -> None:
    """
    Train the model using the features and labels array

    Args:
        features np.ndarray: An numpy array of facial information
        labels np.ndarray: An numpy array of corresponding index of the person.
            This index points to the person in the Key.PEOPLE list
        savepath str: Optionally, a path could be provided to save the trained model
    """
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(features, labels)

    if not savepath:
        savepath = Key.MODEL_FILE

    face_recognizer.save(savepath)

def predict(img: np.ndarray, model: Optional[str]=None) -> List:
    """ Predict the face using the trained model

    Args:
        img np.ndarray: Image in an numpy array format
        model str: Optionally provide your own model file path

    rtype:
        List

    Returns:
        A list of predicted information
        [(face_coords, name, confidence)]
    """
    model = model or Key.MODEL_FILE

    # Reconizer
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read(model)

    predictions = []
    for rect, face in get_faces(img):
        label, confidence = face_recognizer.predict(face)
        predictions.append((rect, Key.PEOPLE[label], confidence))

    return predictions

#-----------------------------------------------------------------------------#
#
# Display the image
#
#-----------------------------------------------------------------------------#
def display_image_with_predictions(img: np.ndarray, predictions: List) -> None:
    """
    Display the image with predicted information: Face rect, predicted name
    and the confidence level

    Args:
        img np.ndarray: An opencv image stored as numpy array
        predictions List: List of face coords, name and confidence
    """
    color = (0,128,0)

    for rect, name, confidence in predictions:
        x,y,w,h = rect
        text = "{0}[{1}%]".format(name.replace("_", " "), int(confidence))

        # Draw a rectangle around the face
        cv.rectangle(img, (x,y), (x+w, y+h), color, thickness=2)
        # Write the name and confidence below the rectangle
        cv.putText(img, text, (x,y+h+20),
                    cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)
    
    show_image(img)

def show_image(img: np.ndarray,
               winname: str= Key.NAME, wait: int= 0) -> None:
    """ Displays the image using cv.imshow method

    Args:
        img np.ndarray: An opencv image stored as numpy array
        winname str: Name of the window
        wait int: Number of milliseconds to wait before the window closes.
            Defaults to 0 and this makes sure the window shows until the
            user closes it.
    """
    cv.imshow(winname, img)
    cv.waitKey(wait)
