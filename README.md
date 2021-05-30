# Layer-Folding

Official implementation of "Layer Folding: Neural Network Depth Reduction using Activation Linearization".

## Introduction
This paper presents Layer Folding: an approach to reduce depth of a pre-trained deep neural network by identifying non-linear activations, such as ReLU, Tanh and Sigmoid, that can be removed, and next by folding the consequent linear layers, i.e. fully connected and convolutional, into one linear layer. The depth reduction can lead to smaller run times especially on edge devices. In addition, it can be shown that some tasks are characterized by so-called “Effective Degree of Non-Linearity (EDNL)”, which hints on how much model non-linear activations can be reduced without heavily compromising the model performance.

## Run Experiments
To install requirements:
```
pip install -r requirements.txt
```

After installation, you can simply execute the script to train and collapse ResNet on Cifar10:
``` python
python ResNet_Cifar10.py
```
The hyper-parameters can be controled by adding arguments. For example:
``` python
python ResNet20_Cifar10.py -d 20 -e 100 -lr 0.001 -m 0.9 -wd 0.0001 -l 0.25
```
Where ```lr``` is a hyperparameter (```λ<sub>c</sub>```) that balances between the task loss and the amount of layers that will be folded, ```d``` is the depth of the net and the rest are set the training process.

Pre-trained models can be found [here](https://github.com/chenyaofo/pytorch-cifar-models).

The following script is folding the activations and then create a shallower network:
``` python
python ResNet20_Cifar10_posfold.py
```
<!--
You can also using the following arguments (all of them not required):

| Short Arg | Long Arg                    | Use                        | Default             |
|:---------:|:---------------------------:|----------------------------|---------------------|
| -e        | --epochs                    | # Epochs                   | 100                 |
| -b        | --batch_size                | Batch Size                 | 128                 |
| -lr       | --learning_rate             | Learning Rate              | 0.001               |
| -m        | --momentum                  | Momentum                   | 0.9                 |
| -wd       | --weight_decay              | Weight Decay               | 1e-4                |
| -l        | --lambda_reg                | Lambda Regularization      | 0.25                |
-->

## Citation

If you find Layer folding method to be useful in your own research, please consider citing the following paper:

```bib
@inproceedings{
BZRA2021,
title={Layer Folding: Neural Network Depth Reduction using Activation Linearization},
author={Amir Ben Dror and Niv Zenghut and Avraham Raviv and Evgeny Artyomov},
booktitle={???},
year={2021},
url={???}
}
```
