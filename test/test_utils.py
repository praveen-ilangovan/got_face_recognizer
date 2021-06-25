import pytest

from src import utils
from . import testdata

def test_file_exists():
    """
    Test if files exist
    """
    for f in testdata.TEST_FILES:
        assert utils.file_exists(f) == True