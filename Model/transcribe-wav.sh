#!/bin/bash

/usr/local/bin/ffmpeg -i $1 -ar 8000 test.wav
rm -rf data/slow-test/output-*.wav
mv test.wav data/slow-test/test.wav
python separate2.py models/slow.yaml
python model.py models/slow.yaml
