# -*- coding: utf-8 -*-
"""
Created on Thu May 28 23:32:51 2020

@author: allen
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import new_dataloader
import torch.utils.data as Data
import matplotlib.pyplot as plt
import test 
import test_profit
import pandas as pd
# Hyper Parameters
EPOCH = 100       # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 1024
INPUT_SIZE = 150         # rnn input size / image width
LR = 0.01               # learning rate

#train_data, train_label, test_data, test_label = dataloader.read_data()
def DataTransfer(train_data, train_label, test_data, test_label):
    
    train_data = torch.FloatTensor(train_data).cuda()
    train_label=torch.LongTensor(train_label).cuda() #data型態轉換@
    test_data=torch.FloatTensor(test_data).cuda()
    test_label=torch.LongTensor(test_label).cuda()
    torch_dataset_train = Data.TensorDataset(train_data, train_label)
    loader_train = Data.DataLoader(
            dataset=torch_dataset_train,      # torch TensorDataset format
            batch_size = BATCH_SIZE,      # mini batch size
            shuffle = True,               
            )

    torch_dataset_test = Data.TensorDataset(test_data, test_label)
    loader_test = Data.DataLoader(
            dataset=torch_dataset_test,      # torch TensorDataset format
            batch_size=BATCH_SIZE,      # mini batch size
            shuffle = False,              
            )
    return loader_train, loader_test

                     # the target label is not one-hotted

#loader_train,loader_test= DataTransfer(train_data, train_label, test_data, test_label)
class CNN_classsification1(nn.Module):
    def __init__(self):
        super(CNN_classsification1,self).__init__()
        #activations=nn.ModuleDict([['ELU',nn.ELU(alpha=1.0)],['ReLU',nn.ReLU()],['LeakyReLU',nn.LeakyReLU()]])
        self.Conv1=nn.Sequential(
                nn.Conv1d(in_channels=3,out_channels=25,kernel_size=5,stride=1,bias=False),
                #nn.Conv1d(25,25, kernel_size=10, stride=1,bias=False),
                nn.BatchNorm1d(25,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
                #activations[activation_function],
                nn.LeakyReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(p=0.5)
                )
        
        self.Conv2=nn.Sequential(
                 nn.Conv1d(25,50,kernel_size=5,stride=1,bias=False),
                 #nn.Conv1d(50,100,kernel_size = 5, stride = 1,bias =False),
                 nn.BatchNorm1d(50,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
                 #activations[activation_function],
                 nn.LeakyReLU(),
                 
                 nn.MaxPool1d(2),
                 nn.Dropout(p=0.3)
                 
                 )   
        
        self.Conv3=nn.Sequential(
                 nn.Conv1d(50,100,kernel_size=5,stride=1,bias=False),
                 nn.BatchNorm1d(100,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
                 #activations[activation_function],
                 nn.LeakyReLU(),
                 nn.MaxPool1d(kernel_size=2),
                 nn.Dropout(p=0.3)
                 )
        self.Conv4=nn.Sequential(
                nn.Conv1d(100,200,kernel_size=5,stride=1,bias=False),
                nn.BatchNorm1d(200,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
                #activations[activation_function],
                nn.LeakyReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(p=0.2)
        
                )
        
        self.classify=nn.Sequential(
                 nn.Linear(in_features=1000,out_features=25,bias=True),
                 #nn.LogSoftmax()
                 )
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.classify(x)
        #print(output)
        return output

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)

def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)



class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:,:,0:-d] + out
        out1 = self.relu(out1)
        # out += residual

        return out1



class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)
        # out += residual

        return out1




class MSResNet(nn.Module):
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=20):
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64
        
        super(MSResNet, self).__init__()
                
        
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 64, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 128, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 256, layers[2], stride=2)
        # self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)

        # maxplooing kernel size: 16, 11, 6
        self.maxpool3 = nn.AvgPool1d(kernel_size=16, stride=1, padding=0)


        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 64, layers[0], stride=2)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, 128, layers[1], stride=2)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, 256, layers[2], stride=2)
        # self.layer5x5_4 = self._make_layer5(BasicBlock5x5, 512, layers[3], stride=2)
        self.maxpool5 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)


        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, 64, layers[0], stride=2)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, 128, layers[1], stride=2)
        self.layer7x7_3 = self._make_layer7(BasicBlock7x7, 256, layers[2], stride=2)
        # self.layer7x7_4 = self._make_layer7(BasicBlock7x7, 512, layers[3], stride=2)
        self.maxpool7 = nn.AvgPool1d(kernel_size=6, stride=1, padding=0)

        # self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(256, num_classes)

        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)


    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
       
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        #print("X0",x0.size())
        
        x = self.layer3x3_1(x0)
        
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)
        
        # x = self.layer3x3_4(x)
        x = self.maxpool3(x)
        """
        y = self.layer5x5_1(x0)
        
        y = self.layer5x5_2(y)
        y = self.layer5x5_3(y)
        # y = self.layer5x5_4(y)
        
        y = self.maxpool5(y)
        
        z = self.layer7x7_1(x0)
    
        z = self.layer7x7_2(z)
        z = self.layer7x7_3(z)
        # z = self.layer7x7_4(z)
        z = self.maxpool7(z)
        """
        #print(x.size())
        #print(y.size())
        #print(z.size())
        out = torch.cat([x], dim=1)

        out = out.squeeze()
        # out = self.drop(out)
        out1 = self.fc(out)

        return out1


# training and testing
def model_train(loader_train,loader_test):
    
    """
    CNN1_class = CNN_classsification1().cuda()
    optimizer = torch.optim.Adam(CNN1_class.parameters(), lr=LR)   # optimize all cnn parameters
    loss_xe = nn.CrossEntropyLoss().cuda()          
    """
    msresnet = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=300) #channel 設定
    msresnet = msresnet.cuda()
    criterion = nn.CrossEntropyLoss(size_average=False).cuda()
    optimizer = torch.optim.Adam(msresnet.parameters(), lr=0.0001,weight_decay= 0.5,amsgrad= True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40,50], gamma=0.1	)
    
    train_loss =[]
    train_acc=[]
    winrate = []
    big_profit  = -10
    save_loss = 1000
    test_acc = 0
    for epoch in range(EPOCH):
            total_train = 0
            correct_train = 0
            action_list=[]
            action_choose=[]
            msresnet.train()
            #scheduler.step()
            
            for step, (batch_x, batch_y) in enumerate(loader_train):   # 分配 batch data, normalize x when iterate train_loader
                output = msresnet(batch_x)               # cnn output
                loss = criterion(output, batch_y)   # mseloss
                
                optimizer.zero_grad()           # clear gradients for this training step
                loss.backward()                 # backpropagation, compute gradients
                optimizer.step()                # apply gradients
                _, predicted = torch.max(output, 1)
                #print("預測分類 :",predicted)
                total_train += batch_y.nelement()
                correct_train += predicted.eq(batch_y).sum().item()
            train_accuracy = 100 * correct_train / total_train 
            train_loss.append(loss.item())                                          
            print('Epoch {}, train Loss: {:.5f}'.format(epoch+1, loss.item()), "Training Accuracy: %.2f %%" % (train_accuracy))
            
            train_acc.append(train_accuracy)

            for step, (batch_x, batch_y) in enumerate(loader_test):
                output = msresnet(batch_x).cuda()
                loss = criterion(output, batch_y).cuda()
                _, predicted = torch.max(output, 1)
                action_choose = predicted.cpu().numpy()
                action_choose = action_choose.tolist()
                action_list.append(action_choose)
            action_list =sum(action_list, [])
            print("幾個 action :",len(action_list))
            #profit = test_profit.reward(action_list)
            profit = 0
            winrate.append(profit)
            #test_accuracy = 100 * correct_test / total_test          #avg_accuracy = train_accuracy / len(train_loader)
            print('Epoch {}, test Loss: {:.5f}'.format(epoch+1, loss.item()), "Testing winrate: %.2f %%" % (profit))
            if train_accuracy >= 0	 :#and profit > big_profit :
                big_profit = profit
                torch.save(msresnet,"test_300(1).pkl")
    draw(train_loss,train_acc,winrate)
def draw(train_loss, train_acc,winrate):
    plt.title('1Res_Net Classificaiton for pair_trading')
    plt.xlabel('Epoch')
    plt.ylabel('Train_loss')
    plt.plot(train_loss)
    plt.show()        
    plt.close()

    plt.title("1Res_Net Classificaiton for pair_trading")
    plt.xlabel('Epoch')
    plt.ylabel('Train_Accuaracy(%)')
    plt.plot(train_acc)
    plt.show()        
    plt.close()
    df = pd.DataFrame({'loss': train_loss,
                   'acc': train_acc,
                   'winrate': winrate})
    df.to_csv("test_300(1).csv",index=False)
#print(action_choose)

if __name__=='__main__':
    choose = 0
    if choose == 0 :
        train_data, train_label, test_data, test_label = new_dataloader.read_data()
        loader_train,loader_test = DataTransfer(train_data, train_label, test_data, test_label)
        model_train(loader_train,loader_test)
    else :
    #model_train()
        test.test_reward()