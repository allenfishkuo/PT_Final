# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:59:37 2020

@author: allen
"""


import numpy as np
import dataloader
import trading_period
import trading_period_by_test
import trading_period_by_gate
import trading_period_by_gate_mean
import os 
import pandas as pd

path_to_average = "C:/Users/Allen/pair_trading DL2/2016/averageprice/"
ext_of_average = "_averagePrice_min.csv"
path_to_minprice = "C:/Users/Allen/pair_trading DL2/2016/minprice/"
ext_of_minprice = "_min_stock.csv"
path_to_compare = "C:/Users/Allen/pair_trading DL2/newstdcompare2016/"
ext_of_compare = "_table.csv"
path_to_python ="C:/Users/Allen/pair_trading DL2"

path_to_choose = "C:/Users/Allen/pair_trading DL/6action Kmeans/"
max_posion = 5
path_to_profit = "C:/Users/Allen/pair_trading DL2/single state/"
def test_testyear():
    total_reward = 0
    total_num = 0
    win_num = 0
    profit_count = 0
    #count_test =[0,0,0,0,0,0]
    #actions =[[1.5, 3]]
    #counts = [0,0,0,0,0]
    datelist = [f.split('_')[0] for f in os.listdir(path_to_compare)]
    #print(datelist)
    #count = 0
    for date in sorted(datelist):
        
        table = pd.read_csv(path_to_compare+date+ext_of_compare)
        mindata = pd.read_csv(path_to_average+date+ext_of_average)
        tickdata = pd.read_csv(path_to_minprice+date+ext_of_minprice)

        #print(tickdata.shape)
        tickdata = tickdata.iloc[166:]
        tickdata.index = np.arange(0,len(tickdata),1)
        
        os.chdir(path_to_python)    
        num = np.arange(0,len(table),1)
        for pair in num: #看table有幾列配對 依序讀入

            #action_choose = 0
            #spread =  table.w1[pair] * np.log(mindata[ str(table.stock1[pair]) ]) + table.w2[pair] * np.log(mindata[ str(table.stock2[pair]) ])
            #spread = spread.T.to_numpy()
            #print(spread)
            
            open,loss = 1.5,5
            profit,opennum,trade_capital ,_  = trading_period_by_gate_mean.pairs( pair ,166,  table , mindata , tickdata , open ,open, loss ,mindata, max_posion , 0.003, 0.01 , 30000000 )
            if profit > 0 and opennum == 1 :
                profit_count +=1
                print("有賺錢的pair",profit)
                print(date)
                print(table.stock1[pair],table.stock2[pair])
                """
                flag = os.path.isfile('C:/Users/Allen/pair_trading DL2/single state/'+str(date)+'_profit.csv')
        
                if not flag :
                    df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"trade_capital":[trade_capital],"open":[open],"loss":[loss],"reward":[profit],"open_num":[opennum]})
                    df.to_csv(path_to_profit+str(date)+'_profit.csv', mode='w',index=False)
                else :
                    df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"trade_capital":[trade_capital],"open":[open],"loss":[loss],"reward":[profit],"open_num":[opennum]})
                    df.to_csv(path_to_profit+str(date)+'_profit.csv', mode='a', header=False,index=False)  
                """
                
                
            elif opennum ==1 and profit < 0 :
                
                print("賠錢的pair :",profit)
                """
                flag = os.path.isfile('C:/Users/Allen/pair_trading DL2/single state/'+str(date)+'_profit.csv')
        
                if not flag :
                    df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"trade_capital":[trade_capital],"open":[open],"loss":[loss],"reward":[profit],"open_num":[opennum]})
                    df.to_csv(path_to_profit+str(date)+'_profit.csv', mode='w',index=False)
                else :
                    df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"trade_capital":[trade_capital],"open":[open],"loss":[loss],"reward":[profit],"open_num":[opennum]})
                    df.to_csv(path_to_profit+str(date)+'_profit.csv', mode='a', header=False,index=False) 
                """
            #print(profit, open_num)
            total_reward += profit
            #print("profit :",profit)
            if profit > 0:
                win_num +=1
            total_num += opennum
            #count_test +=1
            #print(count_test)
    print("利潤  and 開倉次數 and winrate:",total_reward ,total_num,win_num/total_num)
            #print("開倉次數 :",open_num)
test_testyear()