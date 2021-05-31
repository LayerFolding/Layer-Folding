import torch
import torch.utils.data
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from typing import Union, List, Dict, Any, cast
import argparse


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')

parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-wd', '--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-l','--lambda_reg', default=1/4, type=float,
                    metavar='L', help='lambda regularization (default: 1/4)')
parser.add_argument('-d','--depth', default=16, type=int,
                    metavar='L', help='The depth of the network')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
batch_size = 128
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
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=0)


class VGG(nn.Module):
    def __init__(
            self,
            features: nn.Module,
            num_classes: int = 10,
            init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.PReLU(init=0),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.PReLU(init=0),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.PReLU(init=0)]
            else:
                layers += [conv2d, nn.PReLU(init=0)]
            in_channels = v
            '''
        if v != 'M':
            layers[-1].weight.requires_grad = False
            '''
    return nn.Sequential(*layers)

cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg('vgg19', 'E', True, pretrained, progress, **kwargs)


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg('vgg16', 'D', True, pretrained, progress, **kwargs)


if __name__ == "__main__":
    args = parser.parse_args()
    lr = args.learning_rate
    momentum = args.momentum
    lambda_reg = args.lambda_reg
    wd = args.weight_decay
    num_epochs = args.epochs
    depth = args.depth

    if depth == 16:
        model = vgg16()
    elif depth == 19:
        model = vgg19()

    state_dict = torch.load(f'models/vgg{depth}.pt', map_location=device)
    state_dict = {k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    prelus = [module for module in model.modules() if isinstance(module, nn.PReLU)]
    alpha_arr = []
    a_arr = []
    acc_arr = []
    for pr in prelus:
        alpha_arr.append(pr.weight[0])

    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    curr_lr = lr
    optimizer = torch.optim.SGD(model.parameters(), lr=curr_lr,
                          momentum=momentum, weight_decay=wd)

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
                loss += lambda_reg * abs(1 - ai * ai)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Constraint: alpha non negative
            with torch.no_grad():
                for prelu in prelus:
                    prelu.weight.clip_(min=0.)

            if i % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}\n"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        if epoch in [50, 100, 150, 200]:
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
            temp_arr = []
            for a in alpha_arr:
                temp_arr.append(a.item())
            a_arr.append(temp_arr)
            acc_arr.append(100 * correct / total)
    # Save the model checkpoint
    torch.save(model.state_dict(), f'models/prefold_vgg{depth}.ckpt')
    text_file = open(f'models/prefold_vgg{depth}.txt', "w")
    text_file.write(str(correct / total))
    text_file.write("\n")
    text_file.write(str(a_arr))
    text_file.write("\n")
    text_file.write(str(acc_arr))
    text_file.close()
