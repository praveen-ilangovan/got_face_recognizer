import os
import cv2 as cv #type: ignore
import numpy as np
from typing import List

from . import cv_utils, constants, utils

def prepare_training_set() -> None:
    """
    Loop through the images in the training directory and
    prepare features and labels set that could then be passed
    to the train model
    """
    features = []
    labels = []

    for index, person in enumerate(constants.PEOPLE):
        dirpath = os.path.join(constants.TRAINING_DATA_DIR, person)
        for imgname in os.listdir(dirpath):
            imgpath = os.path.join(dirpath, imgname)
            img = cv_utils.read_image(imgpath)
            if not img:
                continue

            for face in cv_utils.get_faces(img):
                # face[0] -> face coordinates
                # face[1] -> grayscale face image
                features.append(face[1])
                labels.append(index)

    np.save(constants.FEATURES_NPY, np.array(features, dtype="object"))
    np.save(constants.LABELS_NPY, np.array(labels))

    print("Training set prepared!")

def train_model() -> None:
    """
    Train the model and save it on disk
    """
    for f in (constants.FEATURES_NPY, constants.LABELS_NPY):
        if not utils.file_exists(f):
            print("Failed to find training set: {0}".format(f))
            print("Preparing training set..")
            prepare_training_set()
            break
    
    features = np.load(constants.FEATURES_NPY, allow_pickle=True)
    labels = np.load(constants.LABELS_NPY)
    cv_utils.train_model(features, labels)

    print("Model trained!")

def who_is_this(imgpath: str, show:bool = True) -> List:
    """
    Extract the face and predict the name of the person using the trained
    model
    """
    if not utils.file_exists(constants.MODEL_FILE):
        print("Trained model not found. Lets train again.")
        train_model()

    img = cv_utils.read_image(imgpath)
    if img is None:
        return []

    predictions = cv_utils.predict(img)    

    if show:
        resized = cv_utils.resize_image(img, height=constants.RESIZE_HEIGHT)
        for rect, name, confidence in predictions:
            x,y,w,h = rect
            text = "{0}[{1}]".format(name.replace("_", " "), int(confidence))
            cv.rectangle(resized, (x,y), (x+w, y+h), (0,128,0), thickness=2)
            cv.putText(resized, text, (x,y+h+20),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,128,0), 1)
        
        cv_utils.show_image(resized)

    return [p[1] for p in predictions]
    