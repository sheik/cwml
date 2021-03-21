#!/bin/bash

rm -rf data/
rm -rf test/
mkdir data
mkdir test
python GenerateData.py
python model.py
