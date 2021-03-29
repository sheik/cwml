# Quick Start

Just run

    ./run.sh models/single.yaml

# How to make your own model

Copy one of the yaml files from models/ and name it. It is in this file you can tweak the parameters.

Once you have your file, you can run it as above, or you can do each step individually:

    bash# python generate-date.py models/quick-example.yaml
    bash# python train-model.py models/quick-example.yaml

And if you have a trained model already and would like to test it again

    bash# python show-model.py models/quick-example.yaml
