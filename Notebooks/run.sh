#!/bin/bash
if [ ! -d venv ]; then
	./buildenv.sh
fi

. venv/bin/activate
./venv/bin/jupyter-notebook Tones.ipynb
