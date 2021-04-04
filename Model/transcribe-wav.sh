#!/bin/bash

/usr/local/bin/ffmpeg -i $1 -ar 8000 test.wav
rm -rf data/single-test/output-*.wav
mv test.wav data/single-test/test.wav
python separate2.py models/single.yaml
python model.py models/single.yaml