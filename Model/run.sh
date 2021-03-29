#!/bin/bash

if [ ! -d venv ]; then
    virtualenv venv
    . venv/bin/activate
    pip install -r requirements.txt
else
    . venv/bin/activate
fi


testdir=$(python get-value.py $1 system.volumes.test)
datadir=$(python get-value.py $1 system.volumes.data)

mkdir data &> /dev/null
mkdir -p $testdir &> /dev/null

rm -rf $testdir/output*.wav
rm -rf $datadir

python test-wav.py $1

mv test.wav $testdir/test.wav
rm -rf $testdir/separate.py
cp separate.py $testdir/
pushd $testdir &> /dev/null
python separate.py
popd &> /dev/null

python generate-data.py $1
python train-model.py $1
