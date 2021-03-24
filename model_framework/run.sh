#!/bin/bash

source venv/bin/activate

python generate-data.py $1
python train-model.py $1
