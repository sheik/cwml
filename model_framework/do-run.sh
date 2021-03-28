#!/bin/bash

rm -rf /mnt/raid/single-test/output*.wav
rm -rf /mnt/raid/single-data

python test-wav.py single.yaml
mv output.wav /mnt/raid/single-test/test.wav
pushd /mnt/raid/single-test
python separate.py
popd
./run.sh single.yaml

