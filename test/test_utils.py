import pytest

from src import utils
from .conftest import TEST_FILES

def test_file_exists():
    """
    Test if files exist
    """
    for f in TEST_FILES:
        assert utils.file_exists(f) == True