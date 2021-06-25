import os
from src import face_recognizer
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
        A list of rectangle coordinates and
        cropped grayscale image of face: [[x,y,w,h], grayscale_img]
    """
    # resize
    img = resize_image(img, height=constants.RESIZE_HEIGHT)

    # convert it to gray
    gray = convert_to_grayscale(img)

    # Detect faces
    cascade_classifier = get_classifer(classifier)
    if cascade_classifier is None:
        return []
    face_rects = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # For each rect, crop the face
    faces = []
    for x,y,w,h in face_rects:
        faces.append( ((x,y,w,h), gray[y:y+h, x:x+w]) )

    return faces

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
    # cascade_file = os.path.join(constants.CLASSIFIERS_DIR, "haar_face.xml")
    # return cv.CascadeClassifier(cascade_file)

    return cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

#-----------------------------------------------------------------------------#
#
# Face Recognizion
#
#-----------------------------------------------------------------------------#
def train_model(features: np.ndarray, labels: np.ndarray, savepath: Optional[str]=None) -> None:
    """
    Train the model using the features and labels array

    Args:
        features np.ndarray: An numpy array of facial information
        labels np.ndarray: An numpy array of corresponding index of the person.
            This index points to the person in the constants.PEOPLE list
        savepath str: Optionally, a path could be provided to save the trained model
    """
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(features, labels)

    if not savepath:
        savepath = constants.MODEL_FILE

    face_recognizer.save(savepath)

def predict(img: Optional[np.ndarray]) -> List:
    """ Predict the face
    """
    # Reconizer
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read(constants.MODEL_FILE)

    predictions = []
    for rect, face in get_faces(img):
        label, confidence = face_recognizer.predict(face)
        predictions.append((rect, constants.PEOPLE[label], confidence))

    return predictions

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
    faces = get_faces(img)

    for rect, face in faces:
        x,y,w,h = rect
        img = resize_image(img, height=constants.RESIZE_HEIGHT)
        cv.rectangle(img, (x,y), (x+w, y+h), (0,128,0), thickness=2)

    show_image(img)