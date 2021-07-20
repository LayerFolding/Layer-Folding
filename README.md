# Layer-Folding

Official implementation of [**"Layer Folding: Neural Network Depth Reduction using Activation Linearization"**.](https://arxiv.org/abs/2106.09309)
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
You can simply execute the script to train and fold ```ResNet``` or ```VGG``` on ```CIFAR10```:
``` python
python ResNet_Cifar10_prefold.py
```
or:

``` python
python VGG_Cifar10_prefold.py
```
The hyper-parameters can be controled by adding arguments. For example:
``` python
python ResNet_Cifar10_prefold.py -d 20 -e 100 -lr 0.001 -m 0.9 -l 0.25
```
Where ```l``` is a hyperparameter that balances between the task loss and the amount of layers that will be folded (```λ```), ```d``` is the depth of the net and the rest set the training process.


The following scripts take a prefold network and then fold its activations, create a shallower network and finally fine-tune the weights:
``` python
python ResNet_Cifar10_posfold.py
```
or:
``` python
python VGG_Cifar10_posfold.py
```
Examples of prefold networks are available in ```models``` directory, for both ```ResNet20``` and ```VGG16```.

**Note**: For training networks on CIFAR100, you should load the relevant datasets:
```
train_dataset = torchvision.datasets.CIFAR100(root='data/', train=True, transform=transform, download=True)

test_dataset = torchvision.datasets.CIFAR100(root='data/', train=False, transform=transforms.Compose([transforms.ToTensor(), normalize,]))
```
And change the number of classes as well:

**ResNet**:
```
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100)
```

**VGG**: 
```
class VGG(nn.Module):
    def __init__(self, features: nn.Module, num_classes: int = 100, init_weights: bool = True)
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
We consider the commonly used MobileNetV2 and EfficientNet-lite. We focus on these models for their attractiveness for hardware and edge devices, mostly credited to their competitive latency and the exclusion of squeeze-and-excite layers employed by other state-of-the-art networks. Folded netwroks and checkpoints can be found under /Folded-Mobilenet directory.
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
Progression of ```α``` values corresponding to non-linear layers in ResNet-20 and ResNet-56 throughout the pre-folding phase with ```λc = 0.25```. As expected, all ```α``` values are either kept around zero or pushed to one.
  
### ResNet20
<p align="center">
  <img width="500" height="300" src="https://user-images.githubusercontent.com/84841423/120116730-31cf7e80-c192-11eb-8280-e1b34ece0a1a.png">
<p align="center">

### ResNet56
<p align="center">
  <img width="500" height="300" src="https://user-images.githubusercontent.com/84841423/120116728-309e5180-c192-11eb-9d3e-85cb0b227aa6.png">
<p align="center">

We also show a table of the indices of the removed layers for folded ResNet and VGG networks such that their resulting depth corresponds to their EDNL. The significant activation in each architecture can be derived from this table:
<p align="center">
  <img width="600" height="375" src="https://user-images.githubusercontent.com/84841423/120116934-5d9f3400-c193-11eb-994a-b7422565d5ce.png">
<p align="center">

## Citation

If you find Layer folding method to be useful in your own research, please consider citing the following paper:

```bib
@inproceedings{
BZRAVJ2021,
title={Layer Folding: Neural Network Depth Reduction using Activation Linearization},
author={Anonymous Author(s)},
year={2021},
url={https://arxiv.org/abs/2106.09309}
}
```
