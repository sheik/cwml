#!/bin/bash
set -x

/usr/local/bin/ffmpeg -i $1 -ar 8000 test.wav &> /dev/null
wpm=$(python wpm-detect.py test.wav)

model=models/single.yaml

if (( $(echo "$wpm > 20.0" |bc -l) )); then
	model=models/fast.yaml
fi

testdir=$(python lib/get-value.py $model system.volumes.test)
datadir=$(python lib/get-value.py $model system.volumes.data)
modeldir=$(python lib/get-value.py $model system.volumes.model)

rm -rf  $testdir/output-*.wav
mv test.wav $testdir/test.wav
python letter-tokenizer.py $model
python model.py $model
