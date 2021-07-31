# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:23:13 2020

@author: allen
"""
import torch
import torch.nn as nn
import numpy as np
import new_dataloader, new_dataloader_return ,new_dataloader_nosp
import MDD

import trading_period_by_gate_mean_new
#import matrix_trading
import os 
import pandas as pd
import torch
import torch.utils.data as Data

import matplotlib.pyplot as plt
import time
path_to_image = "C:/Users/Allen/pair_trading DL/negative profit of 2018/"


path_to_average = "./2018/averageprice/"
ext_of_average = "_averagePrice_min.csv"
path_to_minprice = "./2018/minprice/"
ext_of_minprice = "_min_stock.csv"

path_to_2015compare = "./newstdcompare2015/" 
path_to_2016compare = "./newstdcompare2016/" 
path_to_2017compare = "./newstdcompare2017/" 
path_to_2018compare = "./newstdcompare2018/" 

path_to_2018compare = "./newstdcompare2018/" 

ext_of_compare = "_table.csv"

path_to_python ="C:/Users/Allen/pair_trading DL2"

path_to_half = "C:/Users/Allen/pair_trading DL2/2016/2016_half/"
path_to_2017half = "./2017_halfmin/"
path_to_2018half = "./2018_halfmin/"
ext_of_half = "_half_min.csv"

path_to_profit = "./period_of_train_2/2017_S_P/1_0year/"
max_posion = 5
cost_gate_Train = False
loading_data = False
test_period = {2016 : [path_to_2016compare],2017 :[path_to_2017compare],2018 : [path_to_2018compare]}

time = 2018


def test_reward():
    range_trading_cost_threshold = np.arange(0.0015,0.008,0.0005)
    number_of_label = len(range_trading_cost_threshold)
    total_reward = 0
    total_num = 0
    total_trade = [0,0,0]
    action_list = []
    action_list2 = []
    check = 0
    #actions = [[0.6133408446003465, 20.043565348022334], [0.6557762836185819, 9.672269763651162], [0.7123126470953718, 5.0376563854822525], [1.1433818417759427, 7.290679890624436], [1.4616844455470226, 11.091664599354965], [1.4646842029194989, 18.720335308570633], [1.5278527918781692, 16.150050761421287], [1.8079859466287909, 5.99999999999997], [1.8916951729380467, 12.846636259977194], [1.9903685727286529, 4.999999999999846], [2.030242211571373, 14.706692913385783], [2.1048266865449103, 22.69250838613482], [2.2011242403781104, 9.193923024983164], [2.9250000000000016, 20.10344827586205], [2.927334267040151, 11.825552443199511], [3.254488356362935, 7.52294907720547], [3.508635996771593, 16.51957223567391], [4.2079168858495475, 14.57469752761706], [4.323914893617021, 9.632680851063803], [4.471572794899045, 12.523910733262484], [5.022148337595906, 22.808695652173878], [6.308691275167791, 18.983221476510085], [6.350792751981882, 11.6640241600604], [6.979941239316245, 15.693376068376079], [9.98977440750324, 24.99573317561555]]
    #單純25action 0120
    #actions = [[0.61324904022712, 20.043398430980105], [0.776312126211705, 5.1989069911319366], [0.9245009219914979, 10.470937224404707], [0.9815812088226723, 7.203322830134629], [1.5091524632267055, 22.27597478402984], [1.527852791878169, 16.150050761421287], [1.5375799210991734, 18.736770507414004], [1.5718010456796814, 14.725371491469451], [1.7138786246633644, 8.829335504478017], [1.9849323562570391, 12.502254791431788], [1.9903685727286529, 4.999999999999848], [2.1222512447963604, 6.231409680842454], [2.7909482160211057, 10.338053588782445], [2.9502008928571457, 23.086383928571415], [3.0699452126271876, 8.443777719801744], [3.0746052069995753, 20.23730260349976], [3.209994443672733, 14.640088901236258], [3.5096289574511017, 16.519257914902177], [3.7828111209179194, 12.131067961165044], [3.920916046319274, 7.276054590570744], [5.781510232886389, 10.55422253587393], [5.879370395177493, 22.77494976557263], [6.24432955303536, 14.525683789192806], [6.535820895522393, 18.253731343283597], [10.00000000000039, 24.999999999999815]]
    #去除小於0.1% 在做kmeans
    #actions = [[0.5228608966989476, 10.000000000000071], [0.546959896507091, 6.999999999999915], [0.6132584926132854, 20.043485518737484], [0.6351619919003917, 4.99999999999954], [0.8639708561020019, 9.000000000000012], [1.179748881153653, 6.000000000000028], [1.4149067049415627, 8.000000000000057], [1.4343396226415146, 22.348911465892662], [1.5131965006729389, 11.068640646029628], [1.5651308016877583, 14.734852320675081], [1.598206025047933, 7.000000000000076], [1.614435860582107, 18.70768954365797], [1.6283000902074771, 16.20507166482915], [1.7969422505615502, 5.000000000000077], [1.961350422832984, 13.0], [1.9823461730865428, 9.000000000000075], [2.079203959858973, 5.999999999999988], [2.2296322489391804, 9.999999999999995], [2.395085714285717, 12.000000000000005], [2.630607159039422, 23.2659719075669], [2.785531914893619, 13.999999999999998], [2.8639918116683747, 8.20081883316276], [2.867892644135189, 20.321073558648116], [2.9203545232273824, 15.316975200838316], [9.999999999999979, 24.99999999999971]]
    #去除小於0.5% 在做kmeans
    actions = [[0.5, 10.0], [1.3000000000000007, 23.0], [1.3200000000000008, 6.0], [1.3500000000000008, 7.0], [1.4000000000000008, 20.0], [1.4500000000000008, 5.0], [1.4800000000000008, 11.0], [1.5000000000000009, 5.0],[1.52000000000000009, 8.0], [1.550000000000009, 7.0], [1.6500000000000008, 16.0], [1.7500000000000009, 15.0], [1.8000000000000012, 5.0], [1.8500000000000012, 9.0], [1.9500000000000013, 5.0], [2.0000000000000013, 6.0], [2.1000000000000014, 9.0], [2.200000000000001, 6.0], [2.2500000000000018, 5.0], [2.2500000000000018, 12.0], [2.4000000000000017, 10.0], [2.7500000000000018, 15.0], [2.9000000000000017, 20.0], [3.3500000000000023, 16.0], [10.0, 25.0]]
    #HighFreq 25actions 
    #actions = [[0.5,2.5],[1.0,3.0],[1.5,3.5],[2.0,4.0],[2.5,4.5],[3.0,5.0]] #heuristic 6actions
    #actions = [[0.5, 10.0], [1.3000000000000007, 23.0], [1.4500000000000008, 5.0], [2.1000000000000014, 9.0], [2.2500000000000018, 12.0], [9.881655957107576, 24.92890762493331]]
    
    Net = torch.load("2015-2016_anasgrad_(M3)1.pkl")
    Net.eval()
    #print(Net)
    whole_year = new_dataloader.test_data()
    whole_year = torch.FloatTensor(whole_year).cuda()
    #print(whole_year)
    torch_dataset_train = Data.TensorDataset(whole_year)
    whole_test = Data.DataLoader(
            dataset=torch_dataset_train,      # torch TensorDataset format
            batch_size = 1024,      # mini batch size
            shuffle = False,               
            )
    for step, (batch_x,) in enumerate(whole_test):
        #print(batch_x)
        output = Net(batch_x)               # cnn output
        _, predicted = torch.max(output, 1)
        action_choose = predicted.cpu().numpy()
        action_choose = action_choose.tolist()
        action_list.append(action_choose)
   # action_choose = predicted.cpu().numpy()
    action_list =sum(action_list, [])
    print(len(action_list))
    if cost_gate_Train :
        Model_cost_gate = torch.load('2015-2016training_stucturebreak(3).pkl')

        Model_cost_gate.eval()
        whole_year = new_dataloader.test_data()
        whole_year = torch.FloatTensor(whole_year).cuda()
        torch_dataset_train = Data.TensorDataset(whole_year)
        whole_test = Data.DataLoader(
            dataset=torch_dataset_train,      # torch TensorDataset format
            batch_size = 1024,      # mini batch size
            shuffle = False,               
            )
        for step, (batch_x,) in enumerate(whole_test):
        #print(batch_x)
            output = Model_cost_gate(batch_x)               # cnn output
            _, predicted = torch.max(output, 1)
            action_choose = predicted.cpu().numpy()
            action_choose = action_choose.tolist()
            action_list2.append(action_choose)
        action_list2 =sum(action_list2, [])
    #print(action_list2)

    
    count_test = 0
    datelist = [f.split('_')[0] for f in os.listdir(test_period[time][0])]
    #print(datelist)
    #print(datelist[167:])
    profit_count = 0
    for date in sorted(datelist[:]): #決定交易要從何時開始
        table = pd.read_csv(test_period[time][0]+date+ext_of_compare)
        mindata = pd.read_csv(path_to_average+date+ext_of_average)
        #tickdata = pd.read_csv(path_to_minprice+date+ext_of_minprice)#.drop([266, 267, 268, 269, 270])
        try:
            tickdata = pd.read_csv(path_to_minprice+date+ext_of_minprice).drop([266, 267, 268, 269, 270])
        except:
            continue
        
        #halfmin = pd.read_csv(path_to_half+date+ext_of_half)
        #print(tickdata.shape)
        tickdata = tickdata.iloc[166:]
        #print(tickdata)
        tickdata.index = np.arange(0,len(tickdata),1)  
        num = np.arange(0,len(table),1)
        #print(date)
        for pair in num: #看table有幾列配對 依序讀入
            profit,opennum,trade_capital,trading = 0, 0, 0, [0,0,0]
            #print("action choose :",action_list[count_test])
            for threshold_choose in range(25):
                if action_list[count_test] == threshold_choose :
                    open, loss = actions[threshold_choose][0] , actions[threshold_choose][1] 
                    #open, loss = 2, 1000#
            trading_cost_threshold = 0.0015
            Training = 1
            if cost_gate_Train :
                if action_list2[count_test] == 0 :
                        Training = 0

            if Training == 1 :
                profit,opennum,trade_capital,trading  = trading_period_by_gate_mean_new.pairs( pair ,166,  table , mindata , tickdata , open ,open, loss ,mindata, max_posion , 0.0015, trading_cost_threshold, 300000000 )
            #print(trading)
            if profit > 0 and opennum == 1 :
                profit_count +=1
                #print("有賺錢的pair",profit)
                if loading_data :
                    flag = os.path.isfile(path_to_profit+str(date)+'_profit.csv')
            
                    if not flag :
                        df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"trade_capital":[trade_capital],"open":[open],"loss":[loss],"reward":[profit],"open_num":[opennum]})
                        df.to_csv(path_to_profit+str(date)+'_profit.csv', mode='w',index=False)
                    else :
                        df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"trade_capital":[trade_capital],"open":[open],"loss":[loss],"reward":[profit],"open_num":[opennum]})
                        df.to_csv(path_to_profit+str(date)+'_profit.csv', mode='a', header=False,index=False)  
                    
                    
                
            elif opennum ==1 and profit < 0 :
                
                #print("賠錢的pair :", profit)
                if loading_data :

                    flag = os.path.isfile(path_to_profit+str(date)+'_profit.csv')
                    
                    if not flag :
                        df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"trade_capital":[trade_capital],"open":[open],"loss":[loss],"reward":[profit],"open_num":[opennum]})
                        df.to_csv(path_to_profit+str(date)+'_profit.csv', mode='w',index=False)
                    else :
                        df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"trade_capital":[trade_capital],"open":[open],"loss":[loss],"reward":[profit],"open_num":[opennum]})
                        df.to_csv(path_to_profit+str(date)+'_profit.csv', mode='a', header=False,index=False) 
                    
                    
            #print("開倉次數 :",opennum)
 
            if opennum == 1 or opennum == 0:
                check += 1
                
            total_reward += profit            
            total_num += opennum
            count_test += 1
            total_trade[0] += trading[0]
            total_trade[1] += trading[1]
            total_trade[2] += trading[2]

            
    print("total :",check)        
            #print(count_test)
    print("利潤  and 開倉次數 and 開倉有賺錢的次數/開倉次數:",total_reward ,total_num, profit_count/total_num)
    print("開倉有賺錢次數 :",profit_count)
    print("正常平倉 停損平倉 強迫平倉 :",total_trade[0],total_trade[1],total_trade[2])
    print("正常平倉率 :",total_trade[0]/total_num)
    if loading_data :
        reward,return_reward,per_reward,max_cap ,datelist = MDD.reward_calculation()
        MDD.plot_performance_with_dd(reward,return_reward,per_reward,datelist,total_num,total_trade[0],profit_count/total_num,max_cap )
          
#test()