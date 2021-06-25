# -*- coding: utf-8 -*-

"""
GOT Face Recognizer
Entry point to the module.
"""

# Python built-in imports
import argparse

# Local imports
from . import cv_utils, face_recognizer

#-----------------------------------------------------------------------------#
#
# Arguments
#
#-----------------------------------------------------------------------------#
DES = "Game of Thrones - Face Recognizer: \
An OpenCV python project to detect and recognize characters from \
Game of ThronesTV series."

PARSER = argparse.ArgumentParser(description=DES)
PARSER.add_argument("image",
    help="Provide the filepath to the image that should be recognized.")

#-----------------------------------------------------------------------------#
#
# Main function: Entry point
#
#-----------------------------------------------------------------------------#
def main() -> None:
    """ Main function. Gets called when the module is called from the cmdline.
    """
    args = PARSER.parse_args()
    print(face_recognizer.who_is_this(args.image))

if __name__ == '__main__':
    main()