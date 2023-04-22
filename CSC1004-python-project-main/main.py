from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from utils.config_utils import read_args, load_config, Dict2Object
import time
from multiprocessing import Process,Manager
import multiprocessing
yaxis=["traing_accuracy","training_loss","test_accuracy","test_loss"]
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer):
    """
    tain the model and return the training accuracy
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :return:
    """
    model.train()
    size=len(train_loader.dataset)
    num=size/args.log_interval
    #print(size,num)
    training_accuracy, training_loss=0, 0
    for batch_idx,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        training_loss += loss.item()  
        training_accuracy += (output.argmax(1)==target).type(torch.float).sum().item()

    return  (training_accuracy)/size,training_loss/num


def test(model, device, test_loader,epoch):
    """
    test the model and return the tesing accuracy
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    testing_acc, testing_loss = correct, test_loss  # replace this line
    return testing_acc, testing_loss


def plot(epoches, performance,y_axis):
    plt.cla()
    plt.xlabel("Epoch")
    plt.ylabel(y_axis)
    plt.plot(epoches,performance)
    return plt


def run(config,seed,num,trainAccu,trainLost,testAccu,testLost):
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(seed)

    if use_cuda:
        device = torch.device("cuda")
        print("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("cpu")

    train_kwargs = {'batch_size': config.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': config.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    """add random seed to the DataLoader, pls modify this function"""
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)

    """record the performance"""
    epoches = []
    training_loss = []
    training_accuracies=[]
    testing_accuracies = []
    testing_loss = []
    datay = open("test_data{}.txt".format(num),'a',encoding="utf-8")
    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    for epoch in range(1, config.epochs + 1):
        datax=open("training_data{}.txt".format(num),'a',encoding="utf-8")
        if(epoch==1):
            datax.truncate(0)
        epoches.append(epoch)
        train_auccuracy, train_loss = train(config, model, device, train_loader, optimizer)
        training_accuracies.append(train_auccuracy)
        training_loss.append(train_loss)
        """record training info, Fill your code"""
        test_accuracies, test_loss = test(model, device, test_loader,epoch)
        testing_accuracies.append(test_accuracies/100)
        testing_loss.append(test_loss)
        print(('Epoch:{:2d}, Train_accuracy:{:.2f}%, Train_loss:{:.4f}').format(epoch,100*train_auccuracy,train_loss),file=datax)
        if(epoch==1):
            datay.truncate(0)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, test_accuracies, len(test_loader.dataset),
        100. * test_accuracies / len(test_loader.dataset)),file=datay)
        """record testing info, Fill your code"""
        scheduler.step()
        """update the records, Fill your code"""
    #print(epoches,training_loss,testing_accuracies,testing_loss)
    """plotting training performance with the records"""
    trainAccu.append(training_accuracies)
    trainLost.append(training_loss)
    testAccu.append(testing_accuracies)
    testLost.append(testing_loss)
    plot(epoches, training_loss,yaxis[0]).savefig((r"C:\Users\64703\IdeaProjects\CSC1004-python-project-main\plot\train_loss{}.png").format(num))
    plot(epoches, training_accuracies,yaxis[1]).savefig((r"C:\Users\64703\IdeaProjects\CSC1004-python-project-main\plot\training_accuracy{}.png").format(num))
    """plotting testing performance with the records"""
    plot(epoches, testing_accuracies,yaxis[2]).savefig((r"C:\Users\64703\IdeaProjects\CSC1004-python-project-main\plot\test_accuracy{}.png").format(num))
    plot(epoches, testing_loss,yaxis[3]).savefig((r"C:\Users\64703\IdeaProjects\CSC1004-python-project-main\plot\test_loss{}.png").format(num))

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    manager=Manager()
    trainAccu=manager.list()
    trainLost=manager.list()
    testAccu=manager.list()
    testLost=manager.list()
    time_start=time.time()
    arg = read_args()
    """toad training settings"""
    config = load_config(arg)
    """train model and record results"""
    #Net().share_memory()
    process_list=[]
    p1=Process(target=run,args=(config,123,1,trainAccu,trainLost,testAccu,testLost))
    p1.start()
    process_list.append(p1)
    p2=Process(target=run,args=(config,321,2,trainAccu,trainLost,testAccu,testLost))
    p2.start()
    process_list.append(p2)    
    p3=Process(target=run,args=(config,666,3,trainAccu,trainLost,testAccu,testLost))
    p3.start()
    process_list.append(p3)
    p1.join()
    p2.join()
    p3.join()
    mean_trl=[0.0]*config.epochs
    mean_tra=[0.0]*config.epochs
    mean_tel=[0.0]*config.epochs
    mean_tea=[0.0]*config.epochs
    epoches=[]
    for epoch in range(1, config.epochs + 1):
        epoches.append(epoch)
    for i in range(0,3):
        for j in range(0,config.epochs):
            mean_tra[j]+=(float)(trainAccu[i][j]/3)
            mean_tel[j]+=(float)(testLost[i][j]/3)
            mean_trl[j]+=(float)(trainLost[i][j]/3)
            mean_tea[j]+=(float)(testAccu[i][j]/3)
    #print(testLost)
    #print(epoches)
    plot(epoches, mean_trl,yaxis[0]).savefig((r"C:\Users\64703\IdeaProjects\CSC1004-python-project-main\plot\mean_train_loss.png"))
    plot(epoches, mean_tra,yaxis[1]).savefig((r"C:\Users\64703\IdeaProjects\CSC1004-python-project-main\plot\mean_training_accuracy.png"))
    plot(epoches, mean_tea,yaxis[2]).savefig((r"C:\Users\64703\IdeaProjects\CSC1004-python-project-main\plot\mean_test_accuracy.png"))
    plot(epoches, mean_tel,yaxis[3]).savefig((r"C:\Users\64703\IdeaProjects\CSC1004-python-project-main\plot\mean_test_loss.png"))
    time_end=time.time()
    print('time cost',time_end-time_start,'s')
    
