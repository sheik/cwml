#!/bin/bash

python GenerateData.py $1
python model.py $1
