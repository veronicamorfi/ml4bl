# Machine Learning for Bird Song Learning (ML4BL)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5545932.svg)](https://doi.org/10.5281/zenodo.5545932)

Embedding learning with triplets created from perceptual decisions of birds about song similarity.

zf_embedding_learning:
training (all approaches) and evaluation code.

## Dataset (wavs, melspecs, files):
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5545872.svg)](https://doi.org/10.5281/zenodo.5545872)


## How to use
Python 3 setup: We recommend setting up a Python "virtualenv" before continuing, so that the installed python packages will be stored local to your project rather than system-wide. Then activate the virtualenv and run:

      pip install -r requirements.txt

The Python notebook (`zf_embedding_learning.ipynb`) file shows the full process of training models from triplet data. For this you will need the dataset, available from https://zenodo.org/record/5545872 , and also a computer suitable for deep learning training (e.g. with an NVIDIA GPU and "cuda/cudnn" installed).

To **use** our pretrained model without having to train something new:
The script file `apply_embedding.py` shows a simple example of how to load the model, and apply it to a previously computed mel-spectrogram.

