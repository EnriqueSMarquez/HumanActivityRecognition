from models import resnets,train,msdnet
import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from torch.autograd import Variable
import sys
import utils
import json

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='opp')
parser.add_argument('--downsample_factor',type=int,default=1)
parser.add_argument('--dilated', action='store_true', default=False)
parser.add_argument('--window_size', type=int, default=90)
parser.add_argument('--window_step', type=int, default=45)
parser.add_argument('--saving_folder', type=str, default='')
parser.add_argument('--batch_size', type=int,default=64)
parser.add_argument('--lr',type=float,default=0.01)
parser.add_argument('--momentum',type=float,default=0.9)
parser.add_argument('--nesterov', action='store_true',default=False)
parser.add_argument('--weight_decay', type=float,default=10.e-3)
parser.add_argument('--lr_schedule', type=str, default='100,200')
parser.add_argument('--nb_epochs' , type=int,default=400)
parser.add_argument('--verbose', action='store_true',default=False)
parser.add_argument('--kernel_size',type=int,default=15)
parser.add_argument('--architecture_depth', type=str, default='3,4,4,5')
parser.add_argument('--save_best',action='store_true',default=False)
args = parser.parse_args()

if len(args.saving_folder) == 0:
    args.saving_folder = None

print('LOADING DATA')
datasets = utils.get_datasets(args.dataset,validation=True,window_size=args.window_size,
                              step=args.window_step,downsample=args.downsample_factor)

training_loader = DataLoader(dataset=datasets['training_set'],
                             batch_size=args.batch_size,
                             num_workers=2,
                             drop_last=True,
                             shuffle=True,
                             pin_memory=True)
testing_loader = DataLoader(dataset=datasets['testing_set'],
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=2,
                            drop_last=True,
                            pin_memory=True)
validation_loader = DataLoader(dataset=datasets['validation_set'],
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=2,
                               drop_last=True,
                               pin_memory=True)

if args.saving_folder != None and not os.path.isdir(args.saving_folder):
    os.mkdir(args.saving_folder)

random_training_sample = next(iter(training_loader))[0]
print('Input Shape:')
print(random_training_sample.shape)

model = resnets.HAR_ResNet1D(input_channels=random_training_sample.shape[1],
                             kernel_size=args.kernel_size,
                             depth=[int(item) for item in args.architecture_depth.split(',')],
                             dilated=args.dilated,nb_classes=training_loader.dataset.nb_classes())
model.build_classifier(random_training_sample)
model = model.cuda()

optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,nesterov=bool(args.nesterov),weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()
trainer = train.Trainer(model,training_loader,optimizer,criterion,validation_loader=validation_loader,test_loader=testing_loader,verbose=args.verbose,
                 saving_folder=args.saving_folder,nb_outputs=1,save_best=bool(args.save_best),f1_macro=datasets['mean_metric'])
trainer.train(args.nb_epochs,drop_learning_rate=[int(item) for item in args.lr_schedule.split(',')])

if args.saving_folder != None:
  with open(os.path.join(args.saving_folder,'run_info.json'), 'w') as write_file:
      json.dump(vars(args), write_file)
