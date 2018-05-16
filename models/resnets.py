import torch
from torch import nn
import torchvision
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,kernel_size=3):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size-1)/2)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # we do pre-activation
        out = self.relu(self.bn1(x))
        out = self.conv1(out)

        out = self.relu(self.bn2(out))
        out = self.conv2(out)

        out = self.relu(self.bn3(out))
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class ResNet1D(nn.Module):

    def __init__(self, block, layers, kernel_size=3,input_channels=1):
        self.inplanes = 16
        self.expansion = block.expansion
        self.kernel_size = kernel_size
        super(ResNet1D, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size-1)/2)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0]) #32
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)#16
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)#8
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)#4
        self.bn2 = nn.BatchNorm1d(256*block.expansion)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     nn.Conv2d(self.inplanes, planes * block.expansion,
            #               kernel_size=1, stride=stride),
            #     nn.BatchNorm2d(planes * block.expansion),
            # )

            downsample = nn.Conv1d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,kernel_size=self.kernel_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

def resnet_encoder(depth,block=Bottleneck1D,kernel_size=3,input_channels=1):
    n=int((depth-2)/9)
    model = ResNet1D(block, [n, n, n, n],kernel_size=kernel_size,input_channels=input_channels)
    return model


class HAR_ResNet1D(nn.Module):
    def __init__(self,depth=56,kernel_size=3,input_channels=30,nb_classes=18):
        super(HAR_ResNet1D,self).__init__()
        self.encoder = resnet_encoder(depth,kernel_size=kernel_size,input_channels=input_channels)
        self.last_compression = nn.Conv1d(256*self.encoder.expansion, 16, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)/2)
        self.nb_classes = nb_classes
    def init_weights(self):
        self.encoder.init_weights()
        kaiming_normal_(self.last_compression.weight.data)
        kaiming_normal_(self.classifier.weight.data)
    def build_classifier(self,x):
        x = self.encoder(x)
        x = self.last_compression(x)
        x = x.view(x.size(0),-1)
        self.classifier = nn.Linear(x.size(1),self.nb_classes)
        self.init_weights()
    def forward(self,x):
        x = self.encoder(x)
        x = self.last_compression(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
