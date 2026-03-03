import torch
import torchvision
from torch import nn
from dataset import build_cifar10, build_cifar100
import numpy as np
import torch.optim as optim
from .component54 import decode_one, Net_10, Net_100
from multiprocessing import Pool
import multiprocessing
import multiprocessing.pool
import time


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass


class NoDaemonProcessPool(multiprocessing.pool.Pool):

    def Process(self, *args, **kwds):
        proc = super(NoDaemonProcessPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc




class F_cifar10(object):
    def __init__(self, batch_size=128, epoch = 5 , multi = True, device='cuda:0'):
        self.batch_size=batch_size
        self.epoch=epoch
        self.multi=multi 
        self.device=device
        self.dim=54
        self.device_list=['cuda:4','cuda:5','cuda:6','cuda:7']
        print(self.epoch)

    def build_data(self):
        self.train_loader, self.test_loader=build_cifar10(self.batch_size)

    def cal_num_params(self,net):
        ans=0
        params=net.parameters()
        for param in params:
            ans+=param.numel()
        # print('Net:',ans)
            
        

    def train(self,params,device):
        num_epoch=self.epoch
        train_loader,test_loader=build_cifar10(self.batch_size)
        net=Net_10(*params)
        self.cal_num_params(net)
        # print(net)
        net=net.to(device)
        criterion = nn.CrossEntropyLoss()
        # optimizer=optim.Adam(net.parameters(),lr=lr)
        optimizer = optim.SGD(net.parameters(), lr=0.1,
                        momentum=0.9, nesterov=True, weight_decay=4e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=360)
        
        train_acc_list=[]
        train_loss_traj=[]
        test_acc_list=[]
        test_acc_traj=[]
        
        for epoch in range(num_epoch):
            # print('Epoch :{} on {}'.format(epoch,device))
            net.train()
            # ic(net)
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                # print(net.device)
                # print(inputs.device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                if torch.isnan(loss):
                    return [0.1] * num_epoch , [0.1] * num_epoch
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                train_loss_traj.append(train_loss)

            scheduler.step()
            # print('Lr:',scheduler.get_last_lr())
                    # print('Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(train_loss/(batch_idx+1), 100.*correct/total, correct, total))
                #     print(scheduler.get_lr())
                    

            # print('Train Acc:',100.*correct/total)
            train_acc_list.append(100.*correct/total)

            # eval
            net.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            test_acc_list.append(100.*correct/total)
            # print('Test Acc:', 100.*correct/total)
            if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epoch:
                 print(f"[{device}] Epoch: {epoch+1:3d}/{num_epoch} | Test Acc: {100.*correct/total:.2f}%")
            # print('Test Acc: {:.3f} ({}/{})'.format(100.*correct/total, correct, total))

        return train_acc_list, test_acc_list 

        
    def _call_single(self,params):
        ans=[]
        for item in params:
            ans.append(self.train(*item))
        return ans
            
        


    def _call_multi(self,params):
        
        p=NoDaemonProcessPool(8)

        ans=p.starmap(self.train,params)
        p.close()
        p.join()
        return ans
        
        

    def __call__(self,x,epoch=None):
        # epoch
        if epoch is not None:
            self.epoch=epoch
            

        n,d=x.shape
        params=[]
        # print(n)
        for i in range(n):
            device=self.device_list[i%len(self.device_list)]
            params.append((decode_one(x[i]),device))

        # print(params)


        start_time=time.time()

        if self.multi:
            ans=self._call_multi(params)
        else:
            ans=self._call_single(params)

        end_time=time.time()
        # print(end_time-start_time)
            
        return ans
    


class F_cifar100(object):
    def __init__(self, batch_size=128, epoch = 5 , multi = True, device='cuda:0'):
        self.batch_size=batch_size
        self.epoch=epoch
        self.multi=multi 
        self.device=device
        self.dim=54
        self.device_list=['cuda:2','cuda:3']
        print(self.epoch)

    def build_data(self):
        self.train_loader, self.test_loader=build_cifar100(self.batch_size)

    def cal_num_params(self,net):
        ans=0
        params=net.parameters()
        for param in params:
            ans+=param.numel()
        # print('Net:',ans)
            
        

    def train(self,params,device):
        num_epoch=self.epoch
        train_loader,test_loader=build_cifar100(self.batch_size)
        net=Net_100(*params)
        self.cal_num_params(net)
        # print(net)
        net=net.to(device)
        criterion = nn.CrossEntropyLoss()
        # optimizer=optim.Adam(net.parameters(),lr=lr)
        optimizer = optim.SGD(net.parameters(), lr=0.08,
                        momentum=0.9, nesterov=True, weight_decay=4e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=220)
        
        train_acc_list=[]
        train_loss_traj=[]
        test_acc_list=[]
        test_acc_traj=[]
        
        for epoch in range(num_epoch):
            # print('Epoch :{} on {}'.format(epoch,device))
            net.train()
            # ic(net)
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                # print(net.device)
                # print(inputs.device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                if torch.isnan(loss):
                    return [0.1] * num_epoch , [0.1] * num_epoch
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                train_loss_traj.append(train_loss)

            scheduler.step()
            # print('Lr:',scheduler.get_last_lr())
                    # print('Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(train_loss/(batch_idx+1), 100.*correct/total, correct, total))
                #     print(scheduler.get_lr())
                    

            # print('Train Acc:',100.*correct/total)
            train_acc_list.append(100.*correct/total)

            # eval
            net.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            test_acc_list.append(100.*correct/total)
            # print('Test Acc:', 100.*correct/total)
            if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epoch:
                 print(f"[{device}] Epoch: {epoch+1:3d}/{num_epoch} | Test Acc: {100.*correct/total:.2f}%")
            # print('Test Acc: {:.3f} ({}/{})'.format(100.*correct/total, correct, total))

        return train_acc_list, test_acc_list 

        
    def _call_single(self,params):
        ans=[]
        for item in params:
            ans.append(self.train(*item))
        return ans
            
        


    def _call_multi(self,params):
        
        p=NoDaemonProcessPool(4)

        ans=p.starmap(self.train,params)
        p.close()
        p.join()

        

        return ans
        
        

    def __call__(self,x,epoch=None):
        # epoch
        if epoch is not None:
            self.epoch=epoch
            

        n,d=x.shape
        params=[]
        # print(n)
        for i in range(n):
            device=self.device_list[i%len(self.device_list)]
            params.append((decode_one(x[i]),device))

        # print(params)


        start_time=time.time()

        if self.multi:
            ans=self._call_multi(params)
        else:
            ans=self._call_single(params)

        end_time=time.time()
        # print(end_time-start_time)
            
        return ans
    


if __name__=="__main__":
    x=np.random.randint(0,2,(1,54))
    # x=np.random.randint(0,2,(2,22))
    func=F_cifar10(multi=False,epoch=300)
    ans=func(x)
    print(ans)
        

