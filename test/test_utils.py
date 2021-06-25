# -*- coding: utf-8 -*-

"""
GOT Face Recognizer

Testing src.utils module
"""

# Local imports
from src import utils
from .conftest import (TEST_FILES, INVALID_FILES)

#-----------------------------------------------------------------------------#
#
# Tests
#
#-----------------------------------------------------------------------------#
def test_file_exists():
    """
    Test if files exist
    """
    for f in TEST_FILES:
        assert utils.file_exists(f) == True

def test_file_not_exists():
    """
    Test if files not exist
    """
    for f in INVALID_FILES:
        assert utils.file_exists(f) == False
