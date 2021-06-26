# -*- coding: utf-8 -*-

"""
GOT Face Recognizer

Testing src.constants module
"""

# Python built-in imports
import os

# Local imports
from src import constants as Key

#-----------------------------------------------------------------------------#
#
# Globals
#
#-----------------------------------------------------------------------------#
CWD = os.path.dirname(__file__)
ROOT = os.path.dirname(CWD)

#-----------------------------------------------------------------------------#
#
# Tests
#
#-----------------------------------------------------------------------------#
def test_name():
    assert Key.NAME == "GOT Face Recognizer"

def test_src_directory():
    assert Key.SRC_DIR == os.path.join(ROOT, "src")

def test_resources_directory():
    assert Key.RESOURCES_DIR == os.path.join(ROOT, "resources")

def test_training_directory():
    assert Key.TRAINING_DATA_DIR == os.path.join(ROOT, "resources", "training")
