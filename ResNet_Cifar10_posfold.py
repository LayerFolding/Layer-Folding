'''
This code is modified the official PyTorch implementation:
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
We also used this code:
https://github.com/chenyaofo/pytorch-cifar-models/blob/master/pytorch_cifar_models/resnet.py
There are only 1 main change in the architecture:
The forword process of basic block was changed, using new nn.Module, called Alpha.
To push (some of) alpha(s) from 0 to 1, we also modified the loss function, as the following:
L = L_original + lambda * |[1 - alpha ^ 2]|
All the rest is the same as the original.
'''

import torch
import torch.utils.data
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')

parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-wd', '--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-l','--lambda_reg', default=1/4, type=float,
                    metavar='L', help='lambda regularization (default: 1/4)')
parser.add_argument('-d','--depth', default=20, type=int,
                    metavar='L', help='The depth of the network')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lambda_reg = 1 / 4

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


# Image preprocessing modules
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, 4),
                                transforms.ToTensor(),
                                normalize, ])

# CIFAR-100 dataset
train_dataset = torchvision.datasets.CIFAR10(root='data/', train=True,
                                             transform=transform, download=True)

test_dataset = torchvision.datasets.CIFAR10(root='data/',
                                            train=False,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                normalize,
                                            ]))

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128,
                                           shuffle=True,
                                           num_workers=0)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=128,
                                          shuffle=False,
                                          num_workers=0)


class alpha(nn.Module):
    def __init__(self, alpha_val=0):
        super(alpha, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha_val]).to(device))
        self.alpha.requires_grad = True

    def forward(self, x):
        out = torch.mul(self.alpha, x)
        return out


# convolutions
def conv(in_channels, out_channels, stride=1, kernel=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel//2, bias=bias)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv1 = conv(in_channels, out_channels, stride, kernel=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.conv2 = conv(out_channels, out_channels, kernel=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.alpha1 = alpha(0)
        self.alpha2 = alpha(0)
        self.flag = False
        if stride == 1:
            self.new_conv = conv(in_channels, out_channels, stride, kernel=5, bias=True)
        else:
            self.new_conv = conv(in_channels, out_channels, stride, kernel=7, bias=True)

    def forward(self, x):
        residual = x
        if self.alpha1.alpha.item() < 0.9 or self.flag:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.alpha1(out) + self.activation(out) - self.alpha1(self.activation(out))
            out = self.conv2(out)
        else:
            out = self.new_conv(x)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        if self.alpha2.alpha.item() < 0.9 or self.flag:
            out = self.alpha2(out) + self.activation(out) - self.alpha2(self.activation(out))
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = self.new_conv = conv(3, 16, kernel=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv(self.in_channels, out_channels, stride=stride, kernel=1, bias=False),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == "__main__":

    # set params
    args = parser.parse_args()
    curr_lr = args.learning_rate
    momentum = args.momentum
    lambda_reg = args.lambda_reg
    num_epochs = args.epochs
    nb = int((args.depth - 2) / 6)

    # create the model
    file_name = f'models/prefold_resnet{args.depth}.ckpt'
    model = ResNet(ResidualBlock, [nb, nb, nb]).to(device)
    state_dict = torch.load(file_name)
    alpha_arr = [module for module in model.modules() if isinstance(module, alpha)]
    model.load_state_dict(state_dict, strict=False)

    for a in alpha_arr:
        a.alpha.requires_grad = False

    model = model.to(device)
    model.eval()

    # calculate the new kernels
    index = 0
    dim = 13
    for ml in [model.layer1, model.layer2, model.layer3]:
        for j in range(nb):
            in_c = ml[j].in_channels
            out_c = ml[j].out_channels
            # calculate kernel
            if alpha_arr[index].alpha.item() >= 0.9:
                zeros = torch.zeros([1, in_c, dim, dim], requires_grad=True)
                zeros = zeros.to(device)
                zeros.retain_grad()
                out = ml[j].conv1(zeros)
                out = ml[j].bn1(out)
                out = ml[j].conv2(out)
                c_i, c_j = dim // 2, dim // 2
                if ml[j].stride == 1:
                    biases = out[0, :, dim // 2, dim // 2]
                    Kernel = np.zeros([out_c, in_c, 5, 5])
                    for i in range(out_c):
                        biases[i].backward(retain_graph=True)
                        Kernel[i] = zeros.grad[0, :, c_i - 2:c_i + 3, c_j - 2:c_j + 3].cpu().numpy()
                        zeros.grad.zero_()
                else:
                    biases = out[0, :, dim // 4, dim // 4]
                    Kernel = np.zeros([out_c, in_c, 7, 7])
                    for i in range(out_c):
                        biases[i].backward(retain_graph=True)
                        Kernel[i] = zeros.grad[0, :, c_i - 3:c_i + 4, c_j - 3:c_j + 4].cpu().numpy()
                        zeros.grad.zero_()
                biases = biases.detach().cpu().numpy()
                ml[j].new_conv.weight = nn.Parameter(torch.tensor(Kernel).float())
                ml[j].new_conv.bias = nn.Parameter(torch.tensor(biases).float())
            index += 2
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=curr_lr,
                                momentum=momentum)

    # For updating learning rate
    def update_lr(optimizer_u, lr):
        for param_group in optimizer_u.param_groups:
            param_group['lr'] = lr


    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            for ai in alpha_arr:
                loss += lambda_reg * abs(1 - ai.alpha[0] * ai.alpha[0])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}\n"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # Decay learning rate
        if epoch in [100, 150]:
            curr_lr /= 10
            update_lr(optimizer, curr_lr)

        # Test the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    # Save the model checkpoint
    torch.save(model.state_dict(), f'models/postfold_resnet{args.depth}.ckpt')
    # acc.append(correct / total)
    text_file = open(f"models/postfold_resnet{args.depth}.txt", "w")
    text_file.write(str(correct / total))
    text_file.write("\n")
    alpha_arr_val = []
    alpha_sum = 0
    for ai in alpha_arr:
        alpha_sum += ai.alpha.item()
        alpha_arr_val.append(ai.alpha.item())
    text_file.write(str(alpha_arr_val))
    text_file.write("\n")
    text_file.write(str(alpha_sum))
    text_file.close()
