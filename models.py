import torch
from torch import nn
import torchvision
from torch.autograd import Variable
from torch.nn.init import kaiming_normal
from tqdm import tqdm
import cPickle
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
        self.layer2 = self._make_layer(block, 128, layers[1], stride=self.kernel_size)#16
        self.layer3 = self._make_layer(block, 256, layers[2], stride=self.kernel_size)#8
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)#4
        self.bn2 = nn.BatchNorm1d(256*block.expansion)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)
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
    n=(depth-2)/9
    model = ResNet1D(block, [n, n, n, n],kernel_size=kernel_size,input_channels=input_channels)
    return model

class MLP(nn.Module):
    def __init__(self,input_size,output_size,hidden_units,layers=3):#LEN(HIDDENUNITS) == LAYERS
        super(MLP, self).__init__()
        self.linear_layers = [nn.Sequential(*[nn.Linear(input_size,hidden_units[0]),nn.Dropout(0.5),nn.ReLU()])]
        self.linear_layers += [nn.Sequential(*[nn.Linear(in_features,out_features),nn.Dropout(0.5),nn.ReLU()]) for in_features,out_features in zip(hidden_units[0:-1],hidden_units[1::])]
        self.linear_layers += [nn.Linear(hidden_units[-1],output_size)]
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.init_weights()
    def init_weights(self):##CHECK INIT WEIGHTS COS OF SEQUENTIAL
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_normal(m.weight.data)
            elif isinstance(m, nn.ModuleList):
                for mm in m:
                    if isinstance(mm,nn.Sequential):
                        for mmm in mm:
                            if isinstance(mmm,nn.Linear):
                                kaiming_normal(mmm.weight.data)
            elif isinstance(m,nn.Sequential):
                for mm in m:
                    if isinstance(mm,nn.Linear):
                        kaiming_normal(mm.weight.data)
    def forward(self,x):
        x = x.view(x.size(0),-1)
        for linear_layer in self.linear_layers:
            x = linear_layer(x)
        return x

class HAR_ResNet1D(nn.Module):
    def __init__(self,depth=56,kernel_size=3,input_channels=30):
        super(HAR_ResNet1D,self).__init__()
        self.encoder = resnet_encoder(depth,kernel_size=kernel_size,input_channels=input_channels)
        self.last_compression = nn.Conv1d(256*self.encoder.expansion, 16, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)/2)
        # self.classifier = nn.Linear(16)
        self.init_weights()
    def init_weights(self):
        self.encoder.init_weights()
        kaiming_normal(self.last_compression.weight.data)
    def forward(self,x):
        x = self.encoder(x)
        x = self.last_compression(x)
        return x

class Trainer():
    def __init__(self,model,training_loader,optimizer,criterion,test_loader=None,verbose=True,saving_folder=None):
        self.model = model
        self.verbose = verbose
        self.training_loader = training_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = {'test_acc' : [],
                        'train_acc' : [],
                        'test_loss' : [],
                        'train_loss' : [],
                        'test_f1' : []}
        self.saving_folder = saving_folder
    def train(self,nb_epochs,drop_learning_rate=[]):
        print('TRAINING MODEL WITH EPOCHS %d')%(nb_epochs)
        best_loss = 100.
        starting_epoch = len(self.history['test_loss'])
        for epoch in range(starting_epoch,nb_epochs):
            if epoch in drop_learning_rate:
                for g in self.optimizer.param_groups:
                    g['lr'] = g['lr']*0.1
            if self.verbose:
                print('EPOCH : %d')%(epoch)
            train_loss,train_acc = self.train_epoch()
            if self.test_loader:
                test_loss,test_acc,test_f1 = self.test_epoch()
            self.history['train_loss'] += [train_loss]
            self.history['train_acc'] += [train_acc]
            self.history['test_loss'] += [test_loss]
            self.history['test_acc'] += [test_acc]
            self.history['test_f1'] += [test_f1]
            if self.verbose:
                print('TRAINING ACC : %.4f')%(self.history['train_acc'][-1])
                print('TESTING ACC : %.4f')%(self.history['test_acc'][-1])
                print('TESTING F1 : %.4f')%(self.history['test_f1'][-1])
            self.save_history()
            self.save_model()
            if best_loss > self.history['test_loss'][-1]:
                self.save_model('best_')
    def train_epoch(self):
        total = 0.
        correct = 0.
        running_loss = 0.
        training_loader = tqdm(self.training_loader)
        for i,(x,y) in enumerate(training_loader):
            batch_loss,batch_correct = self.train_batch(x,y)
            total += x.size(0)
            correct += batch_correct
            running_loss = 0.99 * running_loss + 0.01 * batch_loss.data[0]
            training_loader.set_postfix(loss=running_loss)
        return running_loss,correct/total
    def train_batch(self,x,y):
        x = Variable(x).cuda()
        y = Variable(y).cuda().view(-1)
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = self.criterion(out, y)
        _, predicted = torch.max(out.data, 1)
        # _,truth = torch.max(y.data, 1)
        correct = (predicted == y.data).sum()
        loss.backward()
        self.optimizer.step()
        return loss,correct

    def test_epoch(self):
        total = 0.
        correct = 0.
        f1 = []
        running_loss = 0.
        for i,(x,y)  in enumerate(self.test_loader):
            batch_loss,batch_correct,batch_f1 = self.test_batch(x,y)
            running_loss = 0.99 * running_loss + 0.01 * batch_loss.data[0]
            total += x.size(0)
            correct += batch_correct
            f1 += [batch_f1]
        return running_loss,correct/total,np.mean(f1)
    def test_batch(self,x,y):
        x = Variable(x).cuda()
        y = Variable(y).cuda().view(-1)
        out = self.model(x)
        loss = self.criterion(out, y)
        _, predicted = torch.max(out.data, 1)
        # _,truth = torch.max(y.data,1)
        correct = (predicted == y.data).sum()
        f1 = f1_score(y.data.cpu().numpy().reshape(-1),predicted.cpu().numpy().reshape(-1),average='weighted')
        return loss,correct,f1
    def save_history(self):
        with open(self.saving_folder + 'history.txt','w') as fp:
            cPickle.dump(self.history,fp)
    def save_model(self,name=''):
        torch.save(self.model,self.saving_folder+name+'model.pth.tar')
