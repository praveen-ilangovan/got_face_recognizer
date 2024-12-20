# -*- coding: utf-8 -*-

"""
GOT Face Recognizer

Some constants used across the program
"""

# Python built-in imports
import os

NAME = "GOT Face Recognizer"

#-----------------------------------------------------------------------------#
#
# Directories
#
#-----------------------------------------------------------------------------#
SRC_DIR = os.path.dirname(__file__)
RESOURCES_DIR = os.path.join(os.path.dirname(SRC_DIR), "resources")
TRAINING_DATA_DIR = os.path.join(RESOURCES_DIR, "training")
PEOPLE = os.listdir(TRAINING_DATA_DIR)
DATA_DIR = os.path.join(os.path.dirname(SRC_DIR), "data")
MODEL_DIR = os.path.join(os.path.dirname(SRC_DIR), "model")

#-----------------------------------------------------------------------------#
#
# Files
#
#-----------------------------------------------------------------------------#
FEATURES_NPY = os.path.join(DATA_DIR, "features.npy")
LABELS_NPY = os.path.join(DATA_DIR, "labels.npy")
MODEL_FILE = os.path.join(MODEL_DIR, "trained_model.yml")

#-----------------------------------------------------------------------------#
#
# Classifiers
#
#-----------------------------------------------------------------------------#
HAAR_CASCADE = "haar_cascade"
HAAR_CASCADE_FILE = 'haarcascade_frontalface_default.xml'
AVAILABLE_CLASSIFIERS = [HAAR_CASCADE]

#-----------------------------------------------------------------------------#
#
# Face Recognition
#
#-----------------------------------------------------------------------------#
RESIZE_HEIGHT = 250
