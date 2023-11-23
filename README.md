# Breaking the Entanglement of Homophily and Heterophily in Semi-supervised Node Classification

## Overview
This is the official source code of ADPA.

## Requirements
The PyTorch version we use is torch 1.13.0+cu117, and the Optuna version we use is 3.4.0. Please refer to the official website -- https://pytorch.org/get-started/locally/ and https://optuna.org/#installation for the detailed installation instructions.

## Environments
The experiments are conducted on the machine with Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz, and NVIDIA GeForce RTX 3090 with 24GB memory and CUDA 11.8. The operating system is Ubuntu 18.04.6 with 216GB memory.

## Training
Run the following command to test ADPA:
```Training
python3 main_uns_directed_unw.py
```
