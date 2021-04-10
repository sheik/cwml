#!/bin/bash

if [ ! -d venv ]; then
    virtualenv venv
    . venv/bin/activate
    pip install -r requirements.txt
else
    . venv/bin/activate
fi


testdir=$(python lib/get-value.py $1 system.volumes.test)
datadir=$(python lib/get-value.py $1 system.volumes.data)
modeldir=$(python lib/get-value.py $1 system.volumes.model)

mkdir data &> /dev/null
mkdir -p $testdir &> /dev/null

rm -rf $testdir/output*.wav
rm -rf $datadir
rm -rf $modeldir

python test-wav.py $1

mv test.wav $testdir/test.wav

python letter-tokenizer.py $1
python generate-data.py $1
python model.py $1
