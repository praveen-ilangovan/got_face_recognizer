import os

NAME = "GOT Face Recognizer"

# Directories
SRC_DIR = os.path.dirname(__file__)
RESOURCES_DIR = os.path.join(os.path.dirname(SRC_DIR), "resources")
CLASSIFIERS_DIR = os.path.join(RESOURCES_DIR, "classifiers")
TRAINING_DATA_DIR = os.path.join(RESOURCES_DIR, "training")

# Classifiers
HAAR_CASCADE = "haar_cascade"
AVAILABLE_CLASSIFIERS = [HAAR_CASCADE]