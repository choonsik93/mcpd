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
We provide the running code using an example point clouds.

    python train.py \
        --num_mixture 5 \
        --num_iterations 1000 \
        --vis \
        --vis_interval 500 \
        --datadir "data/glasses0" \
        --save \
        --savedir "results/glasses0" \
        --torch

Here's what each option means
* num_mixture - the number of maximum parts
* num_ierations - the number of maximum EM steps
* vis - whether to visualize point clouds during optimization
* vis_interval - if vis is True, it visualize point clouds in every vis_interval EM steps 
* datadir - the path of point clouds data
* save - whether to save the optimized results
* savedir - the path of saved results
* torch - optimization can be done using only numpy when torch is false, and matrix computation can be done using pytorch when torch is true (much faster)

## Basic Usage
Basic usage includes providing any of the registration methods with 2 arrays that are MxN & BxN. E.g., they can have different numbers of points (M & B) but must have the same number of dimensions per point (N).

```python
from mcpd.articulated_registration import ArtRegistration
import numpy as np

# create 3D source point cloud and target point clouds
target = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0], [0, 0.5]])
target = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0], [0, 0.5]])

# set parameters
num_mixture = 5
num_iterations = 1000
vis = False
vis_interval = 1000
torch = True

reg = ArtRegistration(source, target, num_mixture, max_iterations=num_iterations, vis_interval=vis_interval, vis=vis, gpu=torch)
TY, params = reg.register()
```
  
