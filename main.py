import data
import models
import os
import torch
from torch import nn
import numpy as np

def quick_transforms(type=torch.FloatTensor):
    def f(x):
        return torch.from_numpy(x).type(type)
    return f

saving_folder = './Run1_MLP_imbalanced_data/'
if not os.path.isdir(saving_folder):
    os.mkdir(saving_folder)
print(saving_folder)
input_transform = quick_transforms()
target_transform = quick_transforms(type=torch.LongTensor)
training_data = data.OpportunityDataset(dataset='training',transform=input_transform,target_transform=target_transform)
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

model = models.HAR_ResNet1D(input_channels=training_data.data['inputs'].shape[1])
model = model.cuda()
x,y = next(iter(training_data))
x = Variable(x).cuda()
model(x)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=10.e-5)
criterion = nn.CrossEntropyLoss()
trainer = models.Trainer(model,training_loader,optimizer,criterion,test_loader=testing_loader,verbose=True,saving_folder=saving_folder)
trainer.train(500)
