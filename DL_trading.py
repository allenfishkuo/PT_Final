# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:17:19 2020

@author: Allen
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import new_dataloader,new_dataloader_return,new_dataloader_nosp
import torch.utils.data as Data
import matplotlib.pyplot as plt
import test 
import test_profit
import test_dtw
import find_trading_threshold
#import test_cluster
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
import pandas as pd
import pkl_to_pth

rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams['pdf.fonttype'] = 42
# Hyper Parameters
EPOCH = 100  # train the training data n times, to save time, we just train 1 epoch
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
        #print("out5",out.shape)
        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        #print("res5555",residual.shape)
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
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=25):
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
        self.fc = nn.Linear(256*3, num_classes)

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
        #print("input:",x0.shape)
        #print(x0.shape)
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        #print(x0.shape)
        #print("X0",x0.size())
        x = self.layer3x3_1(x0)
        
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)
        
        # x = self.layer3x3_4(x)
        x = self.maxpool3(x)

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
        #print('Z',z.shape)
        #print("x",x.size())
        #print("y",y.size())
        ##print("z",z.size())
        out = torch.cat([x, y, z], dim=1)
        #print(out.shape)
        #print(out.shape)
        out = out.squeeze()
        #print(out.shape)
        # out = self.drop(out)
        out1 = self.fc(out)
        #print(out1.shape)

        return out1


# training and testing
def model_train(loader_train,loader_test):
    
    """
    CNN1_class = CNN_classsification1().cuda()
    optimizer = torch.optim.Adam(CNN1_class.parameters(), lr=LR)   # optimize all cnn parameters
    loss_xe = nn.CrossEntropyLoss().cuda()          
    """
    msresnet = MSResNet(input_channel=3, layers=[1, 1, 1, 1], num_classes=25) #channel 設定
    msresnet = msresnet.cuda()
    criterion = nn.CrossEntropyLoss(size_average=False).cuda()
    optimizer = torch.optim.Adam(msresnet.parameters(), lr=0.001,weight_decay= 0.5,amsgrad= True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40,50], gamma=0.1)
    
    train_loss =[]
    train_acc=[]
    winrate = []
    big_profit  = -100000
    total_profit = []
    for epoch in range(EPOCH):
            total_train = 0
            correct_train = 0
            action_list=[]
            action_choose=[]
            msresnet.train()
            scheduler.step()
            
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
            
            #win,profit = test_profit.reward(action_list) #讓model跑更快
            win,profit = 0,0 
            winrate.append(win)
            total_profit.append(profit)
            #test_accuracy = 100 * correct_test / total_test          #avg_accuracy = train_accuracy / len(train_loader)
            print('Epoch {}, test Loss: {:.5f}'.format(epoch+1, loss.item()), "Testing winrate: %.2f %%" % (profit))
            if train_accuracy >= 50  :
                big_profit = profit
                torch.save(msresnet,"2016-2016input_S_P.pkl")
            
    draw(train_loss,train_acc,winrate,total_profit)
def draw(train_loss, train_acc,winrate,total_profit):
    #plt.title('CNN_Net Classificaiton for pair_trading')
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Train_loss')
    plt.plot(train_loss)
    plt.tight_layout()
    #plt.savefig('Pair Trading ResNet loss(2016).png')
    plt.show()        
    plt.close()

    #plt.title("CNN_Net Classificaiton for pair_trading")
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Train_Accuaracy(%)')
    plt.plot(train_acc)
    plt.tight_layout()
    #plt.savefig('Pair Trading ResNet auc(2016).png')
    plt.show()        
    plt.close()
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('profit(thousands)')
    plt.plot(total_profit)
    plt.tight_layout()
    #plt.savefig('Pair Trading ResNet Validation Winrate(2016).png')
    plt.show()        
    plt.close()
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('winrate (%)')
    plt.plot(winrate)
    plt.tight_layout()
    #plt.savefig('Pair Trading ResNet Validation Winrate(2016).png')
    plt.show()        
    plt.close()
    
    df = pd.DataFrame({'loss': train_loss,
                   'acc': train_acc,
                   'winrate': winrate,
                   "profit" : total_profit})
    #df.to_csv("test300.csv",index=False)

if __name__=='__main__':
    choose = 1
    if choose == 0 :
        train_data, train_label, test_data, test_label = new_dataloader.read_data()
        loader_train,loader_test = DataTransfer(train_data, train_label, test_data, test_label)
        model_train(loader_train,loader_test)
    else :
        #find_trading_threshold.find_trading_cost_threshold()
        test.test_reward()
        #test_dtw.test_reward()
