#!/bin/bash

virtualenv venv
. venv/bin/activate
pip install numpy scipy librosa matplotlib jupyter opencv-python
