# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:31:45 2021

@author: allen
"""
import torch
import torch.nn as nn
import numpy as np
import new_dataloader
import torch.utils.data as Data
import numpy as np
import time
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import trading_period_by_gate_mean_new
import csv
import random
path_to_average = "./2015/averageprice/"
ext_of_average = "_averagePrice_min.csv"
path_to_minprice = "./2015/minprice/"
ext_of_minprice = "_min_stock.csv"
path_to_compare = "./newstdcompare2015/"
ext_of_compare = "_table.csv"


path_to_python ="C:/Users/Allen/PT_Final"
path=os.path.dirname(os.path.abspath(__file__))+'/results/'
path_to_all_gt = "./model/2013-2014_amsgrad_0120(M3)/threshold_label(ST)/"
ext_of_groundtruth = "_ground truth.csv"
max_posion = 5
numbers_of_kmeans = 25
def find_trading_cost_threshold():
    actions = [[0.5, 10.0], [1.3000000000000007, 23.0], [1.3200000000000008, 6.0], [1.3500000000000008, 7.0], [1.4000000000000008, 20.0], [1.4500000000000008, 5.0], [1.4800000000000008, 11.0], [1.5000000000000009, 5.0],[1.52000000000000009, 8.0], [1.550000000000009, 7.0], [1.6500000000000008, 16.0], [1.7500000000000009, 15.0], [1.8000000000000012, 5.0], [1.8500000000000012, 9.0], [1.9500000000000013, 5.0], [2.0000000000000013, 6.0], [2.1000000000000014, 9.0], [2.200000000000001, 6.0], [2.2500000000000018, 5.0], [2.2500000000000018, 12.0], [2.4000000000000017, 10.0], [2.7500000000000018, 15.0], [2.9000000000000017, 20.0], [3.3500000000000023, 16.0], [10.0, 25.0]]

    """actions = [[0.6133408446003465, 20.043565348022334], [0.6557762836185819, 9.672269763651162], [0.7123126470953718, 5.0376563854822525], 
               [1.1433818417759427, 7.290679890624436], [1.4616844455470226, 11.091664599354965], [1.4646842029194989, 18.720335308570633], 
               [1.5278527918781692, 16.150050761421287], [1.8079859466287909, 5.99999999999997], [1.8916951729380467, 12.846636259977194], 
               [1.9903685727286529, 4.999999999999846], [2.030242211571373, 14.706692913385783], [2.1048266865449103, 22.69250838613482], 
               [2.2011242403781104, 9.193923024983164], [2.9250000000000016, 20.10344827586205], [2.927334267040151, 11.825552443199511], 
               [3.254488356362935, 7.52294907720547], [3.508635996771593, 16.51957223567391], [4.2079168858495475, 14.57469752761706], 
               [4.323914893617021, 9.632680851063803], [4.471572794899045, 12.523910733262484], [5.022148337595906, 22.808695652173878], 
               [6.308691275167791, 18.983221476510085], [6.350792751981882, 11.6640241600604], [6.979941239316245, 15.693376068376079], 
               [9.98977440750324, 24.99573317561555]]
    """
    threshold = np.arange(0.0025,0.008,0.0005)
    action_list=[]
    Net = torch.load('2013-2014_amsgrad_0120(M3).pkl')
    Net.eval()
    #print(Net)
    val_year = new_dataloader.find_threshold_data()
    val_year = torch.FloatTensor(val_year).cuda()
    #print(whole_year)
    torch_dataset_train = Data.TensorDataset(val_year)
    val_test = Data.DataLoader(
            dataset=torch_dataset_train,      # torch TensorDataset format
            batch_size = 1024,      # mini batch size
            shuffle = False,               
            )
    for step, (batch_x,) in enumerate(val_test):
        #print(batch_x)
        output = Net(batch_x)               # cnn output
        _, predicted = torch.max(output, 1)
        action_choose = predicted.cpu().numpy()
        action_choose = action_choose.tolist()
        action_list.append(action_choose)
   # action_choose = predicted.cpu().numpy()
    action_list =sum(action_list, [])
    print(len(action_list))
    reward_list=[]
    datelist = [f.split('_')[0] for f in os.listdir(path_to_compare)]
    print(datelist)
    count_test = 0
    for date in sorted(datelist):
        #if date[:6] == '201611' or date[:6] == '201612': 
            table = pd.read_csv(path_to_compare+date+ext_of_compare)
            mindata = pd.read_csv(path_to_average+date+ext_of_average)
            try:
                tickdata = pd.read_csv(path_to_minprice+date+ext_of_minprice).drop([266, 267, 268, 269, 270])
            except:
                continue            
            tickdata = tickdata.iloc[166:]
            tickdata.index = np.arange(0,len(tickdata),1)
            os.chdir(path_to_python)    
            num = np.arange(0,len(table),1)

            for pair in num: #看table有幾列配對 依序讀入
                reward = -1
                open_time = 0 
                open_nn = 0
                threshold_choose = 0
                action_ = 0
                for threshold_choose in range(numbers_of_kmeans):
                    if action_list[count_test] == threshold_choose :
                        open, loss = actions[threshold_choose][0] , actions[threshold_choose][1]
                count_test += 1
                profit,opennum,trade_capital,trading =  trading_period_by_gate_mean_new.pairs( pair ,166,  table , mindata , tickdata , open ,open,loss ,mindata, max_posion , 0.0015, 0.0015 , 300000000 )             
                if profit > 0 :
                    action_ = 1
                else :
                    action_ = 0
                flag = os.path.isfile(path_to_all_gt+'/'+str(date)+'_ground truth.csv')
                    
                if not flag :
                    df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"reward":[profit],"theshold choose":[action_]})
                    df.to_csv(path_to_all_gt+'/'+str(date)+'_ground truth.csv', mode='w',index=False)
                else :
                    df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"reward":[profit],"theshold choose":[action_]})
                    df.to_csv(path_to_all_gt+'/'+str(date)+'_ground truth.csv', mode='a', header=False,index=False)
if __name__ == "__main__":
    find_trading_cost_threshold()