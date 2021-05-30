# Layer-Folding

Official implementation of "Layer Folding: Neural Network Depth Reduction using Activation Linearization".
![LF-1](https://user-images.githubusercontent.com/84841423/120115743-c5528080-c18d-11eb-955b-580c7d0bbda7.png)


## Introduction
This paper presents Layer Folding: an approach to reduce depth of a pre-trained deep neural network by identifying non-linear activations, such as ReLU, Tanh and Sigmoid, that can be removed, and next by folding the consequent linear layers, i.e. fully connected and convolutional, into one linear layer. The depth reduction can lead to smaller run times especially on edge devices. In addition, it can be shown that some tasks are characterized by so-called “Effective Degree of Non-Linearity (EDNL)”, which hints on how much model non-linear activations can be reduced without heavily compromising the model performance.

## Requirements
To install requirements:
```
pip install -r requirements.txt
```

In addition, you should download pre-trained models, and save them in ```models``` directory.

They can be found [here](https://github.com/chenyaofo/pytorch-cifar-models).

## Run Experiments
You can simply execute the script to train and collapse ResNet on Cifar10:
``` python
python ResNet_Cifar10_prefold.py
```
The hyper-parameters can be controled by adding arguments. For example:
``` python
python ResNet_Cifar10_prefold.py -d 20 -e 100 -lr 0.001 -m 0.9 -wd 0.0001 -l 0.25
```
Where ```lr``` is a hyperparameter that balances between the task loss and the amount of layers that will be folded (```λ```), ```d``` is the depth of the net and the rest are set the training process.


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

## Results
We utilize our method to optimize networks with respect to both accuracy and efficiency. 

We perform our experiments on the ImageNet image classification task and measure the latency of all models on NVIDIA Titan X Pascal GPU.
We consider the commonly used MobileNetV2 and EfficientNet-lite. We focus on these models for their attractiveness for hardware and edge devices, mostly credited to their competitive latency and the exclusion of squeeze-and-excite layers employed by other state-of-the-art networks.
<p align="center">
  <img width="460" height="310" src="https://user-images.githubusercontent.com/84841423/120116106-2f1f5a00-c18f-11eb-957d-f29dcd99e591.png">
<p align="center">
  
## Effective Degree of Non-Linearity Evaluation (EDNL)
We use Layer Folding to evaluate the EDNL of several neural networks over image classification tasks.

In order to show that a network possesses an EDNL, we show that its accuracy is roughly maintained down to a certain depth and drops below it. Particularly, we ensure that this holds true even when the network’s size increases as its depth grows smaller. We further show that such depth knee-point is  shared for different networks over a particular task.
### MNIST
<p align="center">
  <img width="460" height="310" src="https://user-images.githubusercontent.com/84841423/120116537-465f4700-c191-11eb-885c-4787cb314cb7.png">
<p align="center">
  
### CIFAR10
<p align="center">
  <img width="460" height="310" src="https://user-images.githubusercontent.com/84841423/120116538-47907400-c191-11eb-96f0-07f8046fb6ed.png">
<p align="center">

### CIFAR100
<p align="center">
  <img width="460" height="310" src="https://user-images.githubusercontent.com/84841423/120116540-48c1a100-c191-11eb-9b20-b345ad08981c.png">
<p align="center">

### Alpha Progression
Progression of ```α``` values corresponding to non-linear layers in ResNet-20 and ResNet-56 throughout the pre-folding phase with λc = 0.25. As expected, all α values are either kept around zero or pushed to one.
  
### ResNet20
<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/84841423/120116730-31cf7e80-c192-11eb-8280-e1b34ece0a1a.png">
<p align="center">

### ResNet56
<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/84841423/120116728-309e5180-c192-11eb-9d3e-85cb0b227aa6.png">
<p align="center">

  
## Citation

If you find Layer folding method to be useful in your own research, please consider citing the following paper:

```bib
@inproceedings{
BZRAVJ2021,
title={Layer Folding: Neural Network Depth Reduction using Activation Linearization},
author={Amir Ben Dror and Niv Zenghut and Avraham Raviv and Evgeny Artyomov},
booktitle={???},
year={2021},
url={???}
}
```
