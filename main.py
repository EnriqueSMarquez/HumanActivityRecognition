from models import resnets,train,msdnet
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
parser.add_argument('--downsample_factor',type=int,default=1)
# parser.add_argument('--dilated',type=bool)
# parser.add_argument('--max_scale',type=int,default=0)
# parser.add_argument('--parallel_index1',type=int)
parser.add_argument('--cross_validation_index',type=int)
# parser.add_argument()
# parser.add_argument('--parallel_index2',type=int)
args = parser.parse_args()
dataset = args.dataset
downsample = int(args.downsample_factor)
cross_validation_index = int(args.cross_validation_index)
if dataset == 'opp':
    from data import opportunity
    dataset = opportunity.OpportunityDataset
elif dataset == 'pamap2':
    from data import pamap2
    dataset = pamap2.PAMAP2_Dataset
elif dataset == 'daph':
    from data import daphnet
    dataset = daphnet.DaphNet
elif dataset == 'hhar':
    from data import hhar
    dataset = hhar.HeterogeneityHumanActivityRecognition()
    dataset.read_data()

# depths_arch = [[[1,1],[1,3],[3,5],[5,7],[9,11],[11,15],[17,19]],
#               [[2,3,4],[3,5,5],[5,7,7],[5,7,9],[7,9,9],[7,9,11],[11,13,15]],
#               [[1,3,3,5],[3,5,5,7],[3,5,7,7],[5,5,7,9],[5,7,9,11],[5,7,9,11],[7,7,9,11]]]
# depths_arch = depths_arch[args.max_scale]
def quick_transforms(type=torch.FloatTensor):
    def f(x):
        return torch.from_numpy(x).type(type)
    return f
saving_folder = './Run2_PAMAP2_CV/'
#Run2_opp
print(saving_folder)

if not os.path.isdir(saving_folder):
    os.mkdir(saving_folder)
print(saving_folder)
# k1 = int(args.parallel_index1)
# k2 = int(args.parallel_index2)
input_transform = quick_transforms()
target_transform = quick_transforms(type=torch.LongTensor)
training_data = dataset(dataset='training',transform=input_transform,
                                        target_transform=target_transform)
test_data = dataset(dataset='test',transform=input_transform,target_transform=target_transform)
nb_classes = training_data.nb_classes
training_data.build_data(window_size=90,step=10,downsample=downsample,cross_validation_index=cross_validation_index)
test_data.build_data(window_size=90,step=10,downsample=downsample,cross_validation_index=cross_validation_index)
# training_sampler = data.BalancedBatchSampler(labels=training_data.data['targets'])


training_loader = torch.utils.data.DataLoader(  dataset=training_data,
                                            batch_size=32,  # specified batch size here
                                            num_workers=2,
                                            drop_last=True,
                                            shuffle=True,  # drop the last batch that cannot be divided by batch_size
                                            pin_memory=True)
                                            # sampler=training_sampler)
testing_loader = torch.utils.data.DataLoader(  dataset=test_data,
                                            batch_size=32,  # specified batch size here
                                            shuffle=True,
                                            num_workers=2,
                                            drop_last=True,  # drop the last batch that cannot be divided by batch_size
                                            pin_memory=True)

# sizes = []
# kernel_sizes = np.arange(3,50,2).astype(int)
# depth_indexs = np.linspace(0,len(depths_arch)-1,len(depths_arch)).astype(int)
# kernel_sizes,depth_indexs = utils.make_grid(kernel_sizes,depth_indexs)
# kernel_size = kernel_sizes.reshape(-1,8)[k1][k2]
# depth = depths_arch[depth_indexs.reshape(-1,8)[k1][k2]]
saving_folder = saving_folder + str(cross_validation_index) +'/'
if not os.path.isdir(saving_folder):
    os.mkdir(saving_folder)
# saving_folder = saving_folder + str(sum(depth)) + '_' + str(kernel_size) +'/'
# if not os.path.isdir(saving_folder):
#     os.mkdir(saving_folder)
model = resnets.HAR_ResNet1D(input_channels=training_data.data['inputs'].shape[1],kernel_size=15,depth=[3,4,4,5],dilated=False,nb_classes=nb_classes)
x = Variable(next(iter(training_loader))[0])
print('Batch shape')
print(x.shape)
# model = msdnet.HAR_MSDNet()
# model.build_classifier(Variable(next(iter(training_loader))[0]))
# model = resnets.HAR_ResNet1D_AuxOuts(input_channels=training_data.data['inputs'].shape[1],kernel_size=17)
model.build_classifier(x)
model = model.cuda()
# optimizer = torch.optim.Adam(model.parameters(),weight_decay=10.e-4)
optimizer = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,nesterov=True,weight_decay=10.e-4)
criterion = nn.CrossEntropyLoss()
trainer = train.Trainer(model,training_loader,optimizer,criterion,test_loader=testing_loader,
                        verbose=True,saving_folder=saving_folder,nb_outputs=1,save_best=True)
trainer.train(400,drop_learning_rate=[10,50,100,200])
print(saving_folder)
