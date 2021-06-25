# Makefile for GOT Face Recognizer

# make test -> to run tests
# make check -> to run mypy
# make run -> to run the module

# Python installer
PYTHON = py

# Runs by default when no target is specified
.DEFAULT_GOAL = all

# Input variable. 
# make run IMAGE="<PATH TO AN IMAGE FILE>"
IMAGE = ""

# TODO: Activate virtual environment

help:
	${PYTHON} -m src --help

check:
	mypy src

tests:
	${PYTHON} -m pytest

run:
	${PYTHON} -m src ${IMAGE}

all: check tests run