# Unsupervised 3D Link Segmentation of Articulated Objects with a Mixture of Coherent Point Drift

This is the code for Unsupervised 3D Link Segmentation of Articulated Objects with a Mixture of Coherent Point Drift.

 * [Project Page](https://choonsik93.github.io/mcpd.github.io)
 * [Paper](https://ieeexplore.ieee.org/document/9790354)
 * [Video](https://www.youtube.com/watch?v=52Rqxs6682A)
 
This codebase is implemented using [CPD](https://github.com/siavashk/pycpd).

## Setup
The code can be run under any environment with Python 3.7 and above.
(It may run with lower versions, but we have not tested it).

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and setting up an environment:

    conda create --name mcpd python=3.7

Next, install the required packages:

    pip install -r requirements.txt

## Running with example data
After preparing a dataset, you can train a Nerfie by running:

    python train.py \
        --data_dir $DATASET_PATH \
        --base_folder $EXPERIMENT_PATH \
        --gin_configs configs/test_vrig.gin
        
To plot telemetry to Tensorboard and render checkpoints on the fly, also
launch an evaluation job by running:

    python eval.py \
        --data_dir $DATASET_PATH \
        --base_folder $EXPERIMENT_PATH \
        --gin_configs configs/test_vrig.gin
