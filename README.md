# PGNets
This repository contains CNN methods to estimate planet mass from protoplanetary disk gaps.

## Credit
This package utilizes planet-disk interactions simulation data in
[Zhang et al. (2018)](https://doi.org/10.3847/2041-8213/aaf744), 
and opacity in 
[Birnstiel et al. (2018)](https://doi.org/10.3847/2041-8213/aaf743).

## Data
The training data is large (2.7G) and not included in this repo.
If you would like to train the model by yourself,
the original data is at available at [here](http://www.physics.unlv.edu/~shjzhang/simimgs.npy).

## Notebooks
Jupyter notebooks are available in the [notebooks folder](notebooks/).

## Dependency

This package uses Tensorflow.
Tensorflow installation
https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/

Install the following package to avoid the died kernel using Jupyter notebook
conda install nomkl
