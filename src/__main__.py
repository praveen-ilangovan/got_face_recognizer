# Python imports
import argparse

# Local imports
from . import cv_utils
from . import constants

#-----------------------------------------------------------------------------#
DES = "Game of Thrones - Face Recognizer: \
An OpenCV python project to detect and recognize characters from Game of \
Thrones TV series."
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
PARSER = argparse.ArgumentParser(description=DES)
PARSER.add_argument("image",
    help="Provide the filepath to the image that should be recognized.")
#-----------------------------------------------------------------------------#


def main() -> None:
    """ Main function. Gets called when the module is called from the cmdline.
    """
    args = PARSER.parse_args()
    cv_utils.display_image(args.image)

if __name__ == '__main__':
    main()