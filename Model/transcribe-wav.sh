#!/bin/bash

/usr/local/bin/ffmpeg -i $1 -ar 8000 -ac 1 test.wav

model=models/single.yaml

if [[ $# -eq 2 ]]; then
	model=$2
else
	wpm=$(python wpm-detect.py test.wav)

	if (( $(echo "$wpm > 20.0" |bc -l) )); then
		model=models/fast.yaml
	fi
fi

testdir=$(python lib/get-value.py $model system.volumes.test)
datadir=$(python lib/get-value.py $model system.volumes.data)
modeldir=$(python lib/get-value.py $model system.volumes.model)

rm -rf  $testdir/output-*.wav
mv test.wav $testdir/test.wav
python letter-tokenizer.py $model
python model.py $model
