import data
from models import resnets,train,msdnet
import os
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import sys

def quick_transforms(type=torch.FloatTensor):
    def f(x):
        return torch.from_numpy(x).type(type)
    return f
saving_folder = './Run1_MSDNet/'
if not os.path.isdir(saving_folder):
    os.mkdir(saving_folder)
print(saving_folder)

# k1 = int(sys.argv[-2])
# k2 = int(sys.argv[-1])

input_transform = quick_transforms()
target_transform = quick_transforms(type=torch.LongTensor)
training_data = data.OpportunityDataset(dataset='training',transform=input_transform,
                                        target_transform=target_transform)
test_data = data.OpportunityDataset(dataset='test',transform=input_transform,target_transform=target_transform)
training_data.build_data(window_size=30,overlap=0.5)
test_data.build_data(window_size=30,overlap=0.5)
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
# kernel_size = np.arange(3,98,2).reshape(-1,4).astype(int)[k1][k2]
# saving_folder = saving_folder + str(kernel_size)+'/'
# if not os.path.isdir(saving_folder):
#     os.mkdir(saving_folder)
# print(saving_folder)
# model = resnets.HAR_ResNet1D(input_channels=training_data.data['inputs'].shape[1])
x = Variable(next(iter(training_loader))[0])
model = msdnet.HAR_MSDNet()
# model.build_classifier(Variable(next(iter(training_loader))[0]))
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0005,weight_decay=10.e-5)
criterion = nn.CrossEntropyLoss()
trainer = train.Trainer(model,training_loader,optimizer,criterion,test_loader=testing_loader,
                        verbose=True,saving_folder=saving_folder,nb_outputs=3)
trainer.train(1000)
