import torch
import torchvision
from torch.autograd import Variable
from tqdm import tqdm
import _pickle as pickle
from sklearn.metrics import f1_score
import numpy as np
import os

class Trainer():
    def __init__(self,model,training_loader,optimizer,criterion,validation_loader=None,test_loader=None,verbose=False,
                 saving_folder=None,nb_outputs=1,save_best=False,f1_macro=True):
        self.model = model
        self.verbose = verbose
        self.training_loader = training_loader
        self.test_loader = test_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.nb_outputs = nb_outputs
        self.f1_macro = f1_macro
        self.histories = [{'test_acc'  : [],
                        'train_acc'  : [],
                        'test_loss' : [],
                        'train_loss' : [],
                        'test_f1' : [],
                        'val_loss' : [],
                        'val_acc' : [],
                        'val_f1': []} for i in range(nb_outputs)]
        self.saving_folder = saving_folder
        self.save_best = save_best
    def train(self,nb_epochs,drop_learning_rate=[],name=''):
        print(('TRAINING MODEL WITH EPOCHS %d')%(nb_epochs))
        best_f1 = 0.
        starting_epoch = len(self.histories[0]['test_loss'])
        for epoch in range(starting_epoch,nb_epochs):
            if epoch in drop_learning_rate:
                for g in self.optimizer.param_groups:
                    g['lr'] = g['lr']*0.1
            print(('EPOCH : %d')%(epoch))
            train_losses,train_accs = self.train_epoch()
            if self.test_loader:
                test_losses,test_accs,test_f1s = self.test_epoch(val=False)
            if self.validation_loader:
                val_losses,val_accs,val_f1s = self.test_epoch(val=True)
            else:
                val_losses,val_accs,val_f1s = test_losses,test_accs,test_f1s
            for i,(history,train_loss,train_acc,test_loss,test_acc,test_f1,val_loss,val_acc,val_f1) in enumerate(zip(self.histories,train_losses,train_accs,
                                                                                            test_losses,test_accs,test_f1s,val_losses,val_accs,val_f1s)):
                self.histories[i]['train_loss'] += [train_loss]
                self.histories[i]['train_acc'] += [train_acc]
                self.histories[i]['test_loss'] += [test_loss]
                self.histories[i]['test_acc'] += [test_acc]
                self.histories[i]['test_f1'] += [test_f1]
                self.histories[i]['val_loss'] += [val_loss]
                self.histories[i]['val_acc'] += [val_acc]
                self.histories[i]['val_f1'] += [val_f1]


            self.print_last_epoch()
            self.save_history(name=name)
            self.save_model(name=name)
            if self.save_best and best_f1 < self.histories[0]['val_f1'][-1]:
                self.save_model('best_')
                best_f1 = self.histories[0]['val_f1'][-1]
                print(('BEST TESTING F1 : %.4f')%(history['test_f1'][-1]))
    def train_epoch(self):
        total = 0.
        corrects = [0.]*self.nb_outputs
        running_losses = [0.]*self.nb_outputs
        if self.verbose:
            training_loader = tqdm(self.training_loader)
        else:
            training_loader = self.training_loader
        for i,(x,y) in enumerate(training_loader):
            batch_losses,batch_corrects = self.train_batch(x,y)
            total += x.size(0)
            corrects = [a + b for a, b in zip(corrects, batch_corrects)]
            running_losses = [0.99 * running_loss + 0.01 * batch_loss.data.item()
                             for running_loss, batch_loss in zip(running_losses, batch_losses)]
            running_losses_dict = {'loss'+str(i) : running_loss for i,running_loss in enumerate(running_losses)}
            if self.verbose:
                training_loader.set_postfix(running_losses_dict)
        corrects = [correct/total for correct in corrects]
        return running_losses,corrects
    def train_batch(self,x,y):
        x = x.cuda()
        y = y.cuda().view(-1)
        self.optimizer.zero_grad()
        outs = self.model(x)
        losses = [self.criterion(out, y) for out in outs]
        corrects = [(torch.max(out, 1)[1] == y.data).sum().item() for out in outs]
        loss = sum(losses) / len(outs)
        loss.backward()
        self.optimizer.step()
        return losses,corrects

    def test_epoch(self,val=False):
        if val:
            test_loader = self.validation_loader
        else:
            test_loader = self.test_loader
        total = 0.
        corrects = [0.]*self.nb_outputs
        running_losses = [0.]*self.nb_outputs
        f1s = []
        for i,(x,y)  in enumerate(test_loader):
            batch_losses,batch_corrects,batch_f1s = self.test_batch(x,y)
            running_losses = [0.99 * running_loss + 0.01 * batch_loss.data.item()
                             for running_loss, batch_loss in zip(running_losses, batch_losses)]
            total += x.size(0)
            corrects = [a + b for a, b in zip(corrects, batch_corrects)]
            f1s += [batch_f1s]
        return running_losses,[correct/total for correct in corrects],np.mean(f1s,axis=0).tolist()
    def test_batch(self,x,y):
        x = x.cuda()
        y = y.cuda().view(-1)
        outs = self.model(x)
        losses = [self.criterion(out, y) for out in outs]
        predicts = [torch.max(out.data, 1)[1] for out in outs]
        corrects = [(torch.max(out.data, 1)[1] == y.data).sum().item() for out in outs]
        if self.f1_macro:
            f1s = [f1_score(y.data.cpu().numpy().reshape(-1),
                            predict.cpu().numpy().reshape(-1),average='macro')
                   for predict in predicts]
        else:
            f1s = [f1_score(y.data.cpu().numpy().reshape(-1),
                            predict.cpu().numpy().reshape(-1),average='weighted')
                   for predict in predicts]
        return losses,corrects,f1s
    def save_history(self,name=''):
        print(('SAVING HISTORY AT %s')%(os.path.join(self.saving_folder, 'history' + name + '.txt')))
        with open(os.path.join(self.saving_folder, 'history' + name + '.txt'),'wb') as fp:
            pickle.dump(self.histories,fp)
    def save_model(self,name=''):
        print(('SAVING MODEL AT %s')%(os.path.join(self.saving_folder,'model' + name + '.pth.tar')))
        torch.save(self.model.state_dict(),os.path.join(self.saving_folder,'model' + name + '.pth.tar'))
    def print_last_epoch(self):
        for i,history in enumerate(self.histories):
            print(('OUTPUT %d')%(i))
            print(('TRAINING ACC : %.4f')%(history['train_acc'][-1]))
            print(('TESTING ACC : %.4f')%(history['test_acc'][-1]))
            print(('TESTING F1 : %.4f')%(history['test_f1'][-1]))
            print(('VAL ACC : %.4f')%(history['val_acc'][-1]))
            print(('VAL F1 : %.4f')%(history['val_f1'][-1]))
    def reset_history(self):
        self.histories = [{'test_acc'  : [],
                        'train_acc'  : [],
                        'test_loss' : [],
                        'train_loss' : [],
                        'test_f1' : []} for i in range(self.nb_outputs)]
