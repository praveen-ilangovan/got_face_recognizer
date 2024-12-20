# -*- coding: utf-8 -*-

"""
GOT Face Recognizer

Generic utilities
"""

# Python built-in imports
import os

def file_exists(path: str) -> bool:
    """
    Checks if the exists on disk

    Args:
        path str: File path

    rtype:
        bool

    Returns:
        True if the file exists
    """
    return os.path.isfile(path)