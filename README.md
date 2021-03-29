# Morse Code Machine Learning 
A machine learning model for CW / morse code

## How to run
1. Download the current [Anaconda](https://www.anaconda.com/products/individual) and install it.  
2. Install virtualenv
```
    $ conda install virtualenv
```
Confirm "yes" when prompted

3.  To run the current notebook, simply download the code and execute the **run.sh** script:
```
    $ cd Model
    $ ./run.sh models/single.yaml
```
# How to make your own model

Copy one of the yaml files from models/ and name it. It is in this file you can tweak the parameters.

Once you have your file, you can run it as above, or you can do each step individually:

    $ python generate-date.py models/quick-example.yaml
    $ python train-model.py models/quick-example.yaml

(C) 2021 Jeff Aigner
