import pytest
import os

from src import constants

CWD = os.path.dirname(__file__)
ROOT = os.path.dirname(CWD)

def test_name():
    assert constants.NAME == "GOT Face Recognizer"

def test_src_directory():
    assert constants.SRC_DIR == os.path.join(ROOT, "src")

def test_resources_directory():
    assert constants.RESOURCES_DIR == os.path.join(ROOT, "resources")

def test_classifiers_directory():
    assert constants.CLASSIFIERS_DIR == os.path.join(ROOT, "resources", "classifiers")

def test_training_directory():
    assert constants.TRAINING_DATA_DIR == os.path.join(ROOT, "resources", "training")

def test_haar_cascade_exists():
    assert os.path.isfile(os.path.join(constants.CLASSIFIERS_DIR, "haar_face.xml"))