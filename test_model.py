from models import resnets,test,msdnet
import os
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import sys
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='opp')
parser.add_argument('--cross_validation_index',type=int)
args = parser.parse_args()
dataset = args.dataset
cross_validation_index = int(args.cross_validation_index)
if dataset == 'opp':
    from data import opportunity
    dataset = opportunity.OpportunityDataset
elif dataset == 'pamap2':
    from data import pamap2
    dataset = pamap2.PAMAP2_Dataset

def quick_transforms(type=torch.FloatTensor):
    def f(x):
        return torch.from_numpy(x).type(type)
    return f
saving_folder = './Run9_pamp2_CV_dilated/' + str(cross_validation_index) + '/'

input_transform = quick_transforms()
target_transform = quick_transforms(type=torch.LongTensor)
test_data = dataset(dataset='test',transform=input_transform,target_transform=target_transform)
test_data.build_data(window_size=90,step=10,cross_validation_index=cross_validation_index)

testing_loader = torch.utils.data.DataLoader(  dataset=test_data,
                                            batch_size=32,  # specified batch size here
                                            shuffle=True,
                                            num_workers=2,
                                            drop_last=True,  # drop the last batch that cannot be divided by batch_size
                                            pin_memory=True)


model = resnets.HAR_ResNet1D(input_channels=test_data.data['inputs'].shape[1],kernel_size=15,depth=[3,4,4,5],dilated=True)
x = Variable(next(iter(testing_loader))[0])
print(x.shape)
model.build_classifier(x)
model.load_state_dict(torch.load(saving_folder+'modelbest_.pth.tar'))
model = model.cuda()
criterion = nn.CrossEntropyLoss()
tester = test.Tester(model,test_loader=testing_loader,criterion=criterion,
                        verbose=True,saving_folder=saving_folder)
tester.test(True)
print(saving_folder)
