import torch
from torch import nn
import torchvision
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,kernel_size=3):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size,padding=(kernel_size-1)/2,stride=stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes,kernel_size=kernel_size,padding=(kernel_size-1)/2)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

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

    def __init__(self, block, layers, kernel_size=3,input_channels=1,return_multiple_outputs=False,first_channels=64):
        self.inplanes = 64
        self.expansion = block.expansion
        self.return_multiple_outputs = return_multiple_outputs
        self.kernel_size = kernel_size
        super(ResNet1D, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size-1)/2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        mid_layers = []
        mid_layers += self._make_layer(block, 64, layers[0]) #32
        channels = first_channels
        for layer in layers[1::]:
            channels *= 2
            mid_layers += self._make_layer(block, channels, layer, stride=2)#16
        self.layers = nn.ModuleList(mid_layers)
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
        outputs = []
        for layer in self.layers:
            outputs += [layer(x)]
            x = outputs[-1]
        if self.return_multiple_outputs:
            return outputs
        else:
            return outputs[-1]

def resnet_encoder(depth,block=BasicBlock1d,kernel_size=3,input_channels=1,return_multiple_outputs=False):
    model = ResNet1D(block, depth,kernel_size=kernel_size,input_channels=input_channels,return_multiple_outputs=return_multiple_outputs)
    return model

class HAR_ResNet1D_AuxOuts(nn.Module):
    def __init__(self,depth=56,kernel_size=3,input_channels=30,nb_classes=18,outputs=3):
        super(HAR_ResNet1D_AuxOuts,self).__init__()
        self.encoder = resnet_encoder(depth,kernel_size=kernel_size,input_channels=input_channels,return_multiple_outputs=True)
        #INCREASE 16, ADD AVERGA POOLING LAYER
        self.last_compressions = [nn.Sequential(nn.BatchNorm1d(int(256*self.encoder.expansion/2**i)),nn.ReLU())]
        self.last_compressions = nn.ModuleList(self.last_compressions)
        self.nb_classes = nb_classes
    def init_weights(self):
        self.encoder.init_weights()
        for compression_container,classifier in zip(self.last_compressions,self.classifiers):
            for m in compression_container.modules():
                if isinstance(m, nn.Conv1d):
                    kaiming_normal_(m.weight.data)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            if isinstance(classifier, nn.Linear):
                kaiming_normal_(classifier.weight.data)
    def build_classifier(self,x):
        x = self.encoder(x)
        self.classifiers = []
        for compression_layer,xx in zip(self.last_compressions,x):
            xx = compression_layer(xx)
            xx = xx.view(xx.size(0),-1)
            self.classifiers += [nn.Linear(xx.size(1),self.nb_classes)]
        self.classifiers = nn.ModuleList(self.classifiers)
        self.init_weights()
    def forward(self,x):
        x = self.encoder(x)
        outputs = []
        for xx,compression,classifier in zip(x,self.last_compressions,self.classifiers):
            xx = compression(xx)
            xx = xx.view(xx.size(0),-1)
            outputs += [classifier(xx)]
        return outputs


class HAR_ResNet1D(nn.Module):
    def __init__(self,depth=[5,5,5,5],kernel_size=5,input_channels=30,nb_classes=18):
        super(HAR_ResNet1D,self).__init__()
        self.encoder = resnet_encoder(depth,kernel_size=kernel_size,input_channels=input_channels)
        self.last_compression = nn.Sequential(nn.BatchNorm1d(int(self.encoder.layers[-1].conv2.out_channels)),nn.ReLU())
        self.nb_classes = nb_classes
    def init_weights(self):
        self.encoder.init_weights()
        for m in self.last_compression.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
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
        return [x]
