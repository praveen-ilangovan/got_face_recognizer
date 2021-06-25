# -*- coding: utf-8 -*-

"""
GOT Face Recognizer

Module that prepares the training set, trains the model and calls the
function that predicts the character.
"""

# Python built-in imports
import os

# Project specific imports
from typing import List
import numpy as np

# Local imports
from . import (cv_utils, utils)
from . import constants as Key

#-----------------------------------------------------------------------------#
#
# Prepare the training set
#
#-----------------------------------------------------------------------------#
def prepare_training_set() -> None:
    """
    Loop through the images in the training directory and
    create features and labels set that could then be passed
    to train model.
    """
    features = []
    labels = []

    for index, person in enumerate(Key.PEOPLE):
        dirpath = os.path.join(Key.TRAINING_DATA_DIR, person)
        for imgname in os.listdir(dirpath):
            img = cv_utils.prepare_image( os.path.join(dirpath, imgname) )
            if not img:
                continue

            for _, face in cv_utils.get_faces(img):
                features.append(face)
                labels.append(index)

    np.save(Key.FEATURES_NPY, np.array(features, dtype="object"))
    np.save(Key.LABELS_NPY, np.array(labels))

    print("Training set prepared!")

#-----------------------------------------------------------------------------#
#
# Train the model
#
#-----------------------------------------------------------------------------#
def train_model() -> None:
    """
    Train the model and save it on disk
    """
    for f in (Key.FEATURES_NPY, Key.LABELS_NPY):
        if not utils.file_exists(f):
            print("Failed to find training set: {0}".format(f))
            print("Preparing training set..")
            prepare_training_set()
            break
    
    features = np.load(Key.FEATURES_NPY, allow_pickle=True)
    labels = np.load(Key.LABELS_NPY)
    cv_utils.train_model(features, labels)

    print("Model trained!")

#-----------------------------------------------------------------------------#
#
# Function that predicts the character
#
#-----------------------------------------------------------------------------#
def who_is_this(imgpath: str, show: bool = True) -> List:
    """
    Extract the face and predict the name of the person using the trained
    model.

    Args:
        imgpath str: Image
        show bool: Displays the image with a rectangle around the face and
            the name of the person. Defaults to True

    rtype:
        List

    Returns:
        A list of names as predicted by the trained model.
    """
    if not utils.file_exists(Key.MODEL_FILE):
        print("Trained model not found. Lets train now.")
        train_model()

    # Read the image
    img = cv_utils.prepare_image(imgpath)
    if img is None:
        return []

    # Get predictions.
    # Returns a list of rect coordinates, name and confidence
    predictions = cv_utils.predict(img)    

    if show:
        cv_utils.display_image_with_predictions(img, predictions)

    # Return the names
    return [p[1] for p in predictions]
    