import data
from models import resnets,train,msdnet
import os
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import sys
import utils

depths_arch = [[1,1],[1,3],[3,5],[5,7],[9,11],
              [1,3,2],[2,3,3],[3,5,5],[5,7,7],[5,7,9],[7,9,9],[7,9,11],
              [1,3,3,1],[3,5,5,5],[3,5,5,7],[3,5,7,7],[5,5,7,9],[5,7,9,11],[5,7,9,11],[7,7,9,11]]

def quick_transforms(type=torch.FloatTensor):
    def f(x):
        return torch.from_numpy(x).type(type)
    return f
# saving_folder = './Run1_Depth_vs_Kernel/'
saving_folder = './Run3_BasicBlock/'

print(saving_folder)

# k1 = int(sys.argv[-2])
# k2 = int(sys.argv[-1])
print(saving_folder)
input_transform = quick_transforms()
target_transform = quick_transforms(type=torch.LongTensor)
training_data = data.OpportunityDataset(dataset='training',transform=input_transform,
                                        target_transform=target_transform)
test_data = data.OpportunityDataset(dataset='test',transform=input_transform,target_transform=target_transform)
training_data.build_data(window_size=90,overlap=0.5)
test_data.build_data(window_size=90,overlap=0.5)
# training_sampler = data.BalancedBatchSampler(labels=training_data.data['targets'])


training_loader = torch.utils.data.DataLoader(  dataset=training_data,
                                            batch_size=64,  # specified batch size here
                                            num_workers=2,
                                            drop_last=True,  # drop the last batch that cannot be divided by batch_size
                                            pin_memory=True)
                                            # sampler=training_sampler)
testing_loader = torch.utils.data.DataLoader(  dataset=test_data,
                                            batch_size=64,  # specified batch size here
                                            shuffle=True,
                                            num_workers=2,
                                            drop_last=True,  # drop the last batch that cannot be divided by batch_size
                                            pin_memory=True)

# sizes = []
# kernel_sizes = np.arange(3,82,2).astype(int)
# depth_indexs = np.linspace(0,len(depths_arch)-1,len(depths_arch)).astype(int)
# kernel_sizes,depth_indexs = utils.make_grid(kernel_sizes,depth_indexs)
# kernel_size = kernel_sizes.reshape(-1,4)[k1][k2]
# depth = depths_arch[depth_indexs.reshape(-1,4)[k1][k2]]
# saving_folder = saving_folder + str(kernel_size) +'_'
# saving_folder = saving_folder + str(depth)[1:-1].split(',')[0]
# for layer_number in str(depth)[1:-1].split(',')[1::]:
#     saving_folder += '-' + layer_number[1]
# saving_folder = saving_folder + '/'
# if not os.path.isdir(saving_folder):
#     os.mkdir(saving_folder)
model = resnets.HAR_ResNet1D(input_channels=training_data.data['inputs'].shape[1],kernel_size=19,depth=[2,3,3])
x = Variable(next(iter(training_loader))[0])
# model = msdnet.HAR_MSDNet()
# model.build_classifier(Variable(next(iter(training_loader))[0]))
# model = resnets.HAR_ResNet1D_AuxOuts(input_channels=training_data.data['inputs'].shape[1],kernel_size=17)
model.build_classifier(x)
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=10.e-3,amsgrad=True)
criterion = nn.CrossEntropyLoss()
trainer = train.Trainer(model,training_loader,optimizer,criterion,test_loader=testing_loader,
                        verbose=False,saving_folder=saving_folder,nb_outputs=1)
trainer.train(500)
print(saving_folder)
