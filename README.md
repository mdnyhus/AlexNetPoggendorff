# AlexNetPoggendorff

This is the directory containing the core code used for my undergraduate honors thesis in Computer Science. I studied AlexNet's perception of the Poggendorff illusion and its ability to differentiate lines from non-lines.

Instructions for running the code:

The AlexNet implementation is in tensorflow and is taken from https://github.com/guerzh/tf_weights. They require an additional file, bvlc_alexnet.npy, which is too large to store on github; it can be downloaded from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/.

[exploratory](./exploratory) - contains code for doing exploratory research. The majority of the python files generate test and control images, and [sim.py](./exploratory/sim.py) and [simPerm.py](./exploratory/simPerm.py) generate .csv files of the similarity scores for these different sets, which are saved in [out](./exploratory/out). This folder also contains the analysis folders.

[lines.py](lines.py) - this is the file used to generate image sets. It takes a number of command line arguments, which can be learned by running `python lines.py -h`.

[AlexNet.py](AlexNet.py) - this is the file used to retrain AlexNet. It takes a number of command line arguments, which can be learned by running `python AlexNet.py -h`.

[analysis.py](analysis.py) - this file is used to generate an analysis .txt file and different charts. It looks for .out files in an ./out subdirectory that contains the command line print statements from running AlexNet.py.
