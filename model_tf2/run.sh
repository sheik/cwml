#!/bin/bash

python generate-data.py $1
python train-model.py $1
