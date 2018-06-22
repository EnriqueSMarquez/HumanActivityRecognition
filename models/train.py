import torch
import torchvision
from torch.autograd import Variable
from tqdm import tqdm
import _pickle as pickle
from sklearn.metrics import f1_score
import numpy as np

class Trainer():
    def __init__(self,model,training_loader,optimizer,criterion,test_loader=None,verbose=False,saving_folder=None,nb_outputs=1,save_best=False):
        self.model = model
        self.verbose = verbose
        self.training_loader = training_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.nb_outputs = nb_outputs
        self.histories = [{'test_acc'  : [],
                        'train_acc'  : [],
                        'test_loss' : [],
                        'train_loss' : [],
                        'test_f1' : []} for i in range(nb_outputs)]
        self.saving_folder = saving_folder
        self.save_best = save_best
    def cross_validate_training(self,nb_epochs,drop_learning_rate=[],nb_files=None,window_size=90,step=10):
        if nb_files == None:
            nb_files = len(self.training_loader.dataset.files)
        for cross_validation_index in range(nb_files):
            print(('TRAINING CROSSVALIDATION INDEX %d')%(cross_validation_index))
            self.training_loader.dataset.build_data(window_size=90,step=10,cross_validation_index=cross_validation_index)
            self.test_loader.dataset.build_data(window_size=90,step=10,cross_validation_index=cross_validation_index)
            self.model.init_weights()
            self.reset_history()
            self.train(nb_epochs=nb_epochs,drop_learning_rate=drop_learning_rate,name='_index'+str(cross_validation_index))
    def train(self,nb_epochs,drop_learning_rate=[],name=''):
        print(('TRAINING MODEL WITH EPOCHS %d')%(nb_epochs))
        best_loss = 100.
        starting_epoch = len(self.histories[0]['test_loss'])
        for epoch in range(starting_epoch,nb_epochs):
            if epoch in drop_learning_rate:
                for g in self.optimizer.param_groups:
                    g['lr'] = g['lr']*0.1
            print(('EPOCH : %d')%(epoch))
            train_losses,train_accs = self.train_epoch()
            if self.test_loader:
                test_losses,test_accs,test_f1s = self.test_epoch()
            # self.history['train_loss'] += [train_loss]
            # self.history['train_acc'] += [train_acc]
            # self.history['test_loss'] += [test_loss]
            # self.history['test_acc'] += [test_acc]
            # self.history['test_f1'] += [test_f1]
            for i,(history,train_loss,train_acc,test_loss,test_acc,test_f1) in enumerate(zip(self.histories,train_losses,train_accs,test_losses,test_accs,test_f1s)):
                self.histories[i]['train_loss'] += [train_loss]
                self.histories[i]['train_acc'] += [train_acc]
                self.histories[i]['test_loss'] += [test_loss]
                self.histories[i]['test_acc'] += [test_acc]
                self.histories[i]['test_f1'] += [test_f1]

            for i,history in enumerate(self.histories):
                print(('OUTPUT %d')%(i))
                print(('TRAINING ACC : %.4f')%(history['train_acc'][-1]))
                print(('TESTING ACC : %.4f')%(history['test_acc'][-1]))
                print(('TESTING F1 : %.4f')%(history['test_f1'][-1]))
            self.save_history(name=name)
            # self.save_model(name=name)
            if self.save_best and best_loss > self.history['test_loss'][-1]:
                self.save_model('best_')
    def train_epoch(self):
        total = 0.
        corrects = [0.]*self.nb_outputs
        running_losses = [0.]*self.nb_outputs
        if self.verbose:
            training_loader = tqdm(self.training_loader)
        else:
            training_loader = self.training_loader
        # training_loader = self.training_loader
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
        x = Variable(x).cuda()
        y = Variable(y).cuda().view(-1)
        self.optimizer.zero_grad()
        outs = self.model(x)
        losses = [self.criterion(out, y) for out in outs]
        # _, predicted = torch.max(out.data, 1)
        # _,truth = torch.max(y.data, 1)
        corrects = [(torch.max(out, 1)[1] == y.data).sum().item() for out in outs]
        # [loss.backward(retain_graph=True) for loss in losses[0:-1]]
        # losses[-1].backward()
        loss = sum(losses) / len(outs)
        loss.backward()
        self.optimizer.step()
        return losses,corrects

    def test_epoch(self):
        total = 0.
        corrects = [0.]*self.nb_outputs
        running_losses = [0.]*self.nb_outputs
        f1s = []
        for i,(x,y)  in enumerate(self.test_loader):
            batch_losses,batch_corrects,batch_f1s = self.test_batch(x,y)
            running_losses = [0.99 * running_loss + 0.01 * batch_loss.data.item()
                             for running_loss, batch_loss in zip(running_losses, batch_losses)]
            total += x.size(0)
            corrects = [a + b for a, b in zip(corrects, batch_corrects)]
            f1s += [batch_f1s]
        return running_losses,[correct/total for correct in corrects],np.mean(f1s,axis=0).tolist()
    def test_batch(self,x,y):
        x = Variable(x).cuda()
        y = Variable(y).cuda().view(-1)
        outs = self.model(x)
        losses = [self.criterion(out, y) for out in outs]
        predicts = [torch.max(out.data, 1)[1] for out in outs]
        # _,truth = torch.max(y.data,1)
        corrects = [(torch.max(out.data, 1)[1] == y.data).sum().item() for out in outs]
        f1s = [f1_score(y.data.cpu().numpy().reshape(-1),
                        predict.cpu().numpy().reshape(-1),average='weighted')
               for predict in predicts]
        return losses,corrects,f1s
    def save_history(self,name=''):
        print(('SAVING HISTORY AT %s')%(self.saving_folder + 'history' + name + '.txt'))
        with open(self.saving_folder + 'history' + name + '.txt','wb') as fp:
            pickle.dump(self.histories,fp)
    def save_model(self,name=''):
        torch.save(self.model.state_dict(),self.saving_folder+'model' + name + '.pth.tar')
    def reset_history(self):
        self.histories = [{'test_acc'  : [],
                        'train_acc'  : [],
                        'test_loss' : [],
                        'train_loss' : [],
                        'test_f1' : []} for i in range(self.nb_outputs)]
