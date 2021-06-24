import argparse

#-----------------------------------------------------------------------------#
DES = "Game of Thrones: Face Recognizer\n\
An OpenCV python project to detect and recognize characters\
from Game of Thrones TV series."
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
    print(args)

if __name__ == '__main__':
    main()