# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 22:50:39 2020

@author: allen
"""

import numpy as np
import time
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import trading_period_by_gate_mean_new
import csv
import random
path_to_average = "./2018/averageprice/"
ext_of_average = "_averagePrice_min.csv"
path_to_minprice = "./2018/minprice/"
ext_of_minprice = "_min_stock.csv"
path_to_compare = "./newstdcompare2015/"
ext_of_compare = "_table.csv"


path_to_python ="C:/Users/Allen/PT_Final"
path=os.path.dirname(os.path.abspath(__file__))+'/results/'
path_to_all_gt = "./gt_cost_gate/"
ext_of_groundtruth = "_ground truth.csv"
max_posion = 5


reward_list=[]

lower_bound = np.arange(0.5,8,0.05)
upper_bound = np.arange(5,25,1)
"""
def choose_action(lower_bound,upper_bound) :
    action_list=[]
    count = 0
    l , u = 1,0
    while count < 300 :
        l = np.random.choice(lower_bound,1)
        u = np.random.choice(upper_bound,1)        
        if 1.5*l < u :
            w = np.concatenate((l,u),axis = None)
            w = list(w)
            #print(w)
            action_list.append(w)
            count +=1
    return action_list
    
action_list = choose_action(lower_bound, upper_bound)
actions = sorted(action_list, key = lambda s: s[0])
print(actions)
"""
# of > 1sigma && delete < 300
actions = [[1.2051618958235575, 22.56405443453776], [1.231789137380196, 6.000000000000029], [1.359038901601831, 14.33695652173912], [1.36092703560099, 11.462812830454698], [1.4000000000000115, 20.000000000000014], [1.4083888426311382, 7.00000000000001], [1.5758556891766924, 5.000000000000011], [1.6838230972671167, 19.0], [1.7218118582312045, 16.247432924809562], [1.8085359544749187, 8.000000000000036], [1.850862214708367, 9.00000000000001], [2.1578632478632525, 6.000000000000065], [2.272758877576358, 5.000000000000066], [2.432984073763622, 22.00000000000002], [2.6697396500213415, 15.000000000000028], [2.8487746275828947, 9.999999999999991], [2.9000000000000012, 20.00000000000002], [2.972463077656029, 8.19580752739403], [2.981865912762522, 12.0], [3.0402494908350324, 7.0000000000000195], [3.041631711409398, 16.238674496644304], [3.1233548387096803, 23.51096774193548], [3.170312500000002, 13.999999999999993], [7.8999999999999995, 20.00000000000011], [7.949999999999917, 19.000000000000053]]
print(len(actions))
def check_cost_gate():
    for date in sorted(datelist):
        table = pd.read_csv(path_to_compare+date+ext_of_compare)
        mindata = pd.read_csv(path_to_average+date+ext_of_average)
        tickdata = pd.read_csv(path_to_minprice+date+ext_of_minprice)
        tickdata = tickdata.iloc[166:]
        tickdata.index = np.arange(0,len(tickdata),1)
        #gt = pd.read_csv(path_to_choose2018+date+ext_of_groundtruth,usecols=["action choose"])
        #gt = gt.values
        #gt= np.array(gt)
        #gt= gt.ravel()
        
        
        os.chdir(path_to_python)    
        num = np.arange(0,len(table),1)
        #gt = gt.ravel()
       # print(gt[0][0])
        for pair in num: #看table有幾列配對 依序讀入
           # print(pair)
            earn_open={}
            reward = -0.000001
            open_time = 0 
            loss_time = 0
            open_nn =0
            action_choose = []
            #spread =  table.w1[pair] * np.log(mindata[ str(table.stock1[pair]) ]) + table.w2[pair] * np.log(mindata[ str(table.stock2[pair]) ])
            #spread = spread.T.to_numpy()
            #print(spread)
            Bth1 = np.ones((5,1))
            #print(tickdata[str(table.stock1[pair])])
            #TickTP1 = tickdata[[str(table.stock1[pair]),str(table.stock2[pair])]]
            #TickTP1 = TickTP1.T.to_numpy()
            #print(TickTP1)
            #choose = int(gt[pair])
            #open,loss = actions[gt[pair][0]]
            for open,loss in sorted(actions): #對不同開倉門檻做測試
                #print(open,loss)
                
                #Bth1[2][0] = table.mu[pair]
                #Bth1[0][0] = table.mu[pair] +table.stdev[pair]*loss
                #Bth1[1][0] = table.mu[pair] +table.stdev[pair]*open
                #Bth1[3][0] = table.mu[pair] -table.stdev[pair]*open
                #Bth1[4][0] = table.mu[pair] -table.stdev[pair]*loss
                #print(Bth1)
                profit,opennum,trade_capital,trading =  trading_period_by_gate_mean_new.pairs( pair ,166,  table , mindata , tickdata , open ,open, 1000 ,mindata, max_posion , 0.0015, 0 , 300000000 )
                #spread ,profit ,opennum, rt = trading_period.pairs( pair ,  table , mindata , tickdata , open , loss , max_posion , 0 , 30000000 )
                #print("利潤 :",profit)
                #print("開倉次數 :",open_num)
                action_choose.append(profit)
    
            """
                plt.figure()
                plt.plot(spread)
                plt.plot(range(len(spread)),plotB1.T,"--")
                plt.title("Trade open :"+str(open)+"and loss :"+str(loss)+" stock :"+str(table.stock1[pair]) +" with "+str(table.stock2[pair]))
                #TC1fn = "Number"+str(self.count_pic)+"Pair Trading RL.png"
                #plt.savefig(path+TC1fn)
                plt.show()
                plt.close()
            """ 
           # print(date[0:4]) 
        #print("====================================================================================")
            flag = os.path.isfile(path_to_all_gt+str(date[0:4])+'/'+str(date)+'_ground truth.csv')
            
            if not flag :
                df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"action1":[action_choose[0]],"action2":[action_choose[1]],"action3":[action_choose[2]],"action4":[action_choose[3]],
                                   "action5":[action_choose[4]],"action6":[action_choose[5]],"action7":[action_choose[6]],"action8":[action_choose[7]],"action9":[action_choose[8]],
                                   "action10":[action_choose[9]],"action11":[action_choose[10]],"action12":[action_choose[11]],"action13":[action_choose[12]],"action14":[action_choose[13]],"action15":[action_choose[14]],
                                   "action16":[action_choose[15]],"action17":[action_choose[16]],"action18":[action_choose[17]],"action19":[action_choose[18]],"action20":[action_choose[19]],
                                   "action21":[action_choose[20]],"action22":[action_choose[21]],"action23":[action_choose[22]],"action24":[action_choose[23]],"action25":[action_choose[24]]})
                df.to_csv(path_to_all_gt+str(date[0:4])+'/'+str(date)+'_ground truth.csv', mode='w',index=False)
            else :
                df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"action1":[action_choose[0]],"action2":[action_choose[1]],"action3":[action_choose[2]],"action4":[action_choose[3]],
                                   "action5":[action_choose[4]],"action6":[action_choose[5]],"action7":[action_choose[6]],"action8":[action_choose[7]],"action9":[action_choose[8]],
                                   "action10":[action_choose[9]],"action11":[action_choose[10]],"action12":[action_choose[11]],"action13":[action_choose[12]],"action14":[action_choose[13]],"action15":[action_choose[14]],
                                   "action16":[action_choose[15]],"action17":[action_choose[16]],"action18":[action_choose[17]],"action19":[action_choose[18]],"action20":[action_choose[19]],
                                   "action21":[action_choose[20]],"action22":[action_choose[21]],"action23":[action_choose[22]],"action24":[action_choose[23]],"action25":[action_choose[24]]})
                df.to_csv(path_to_all_gt+str(date[0:4])+'/'+str(date)+'_ground truth.csv', mode='a', header=False,index=False)
                #print(P1)
            #print(C1)
def profit_distribution(datelist):
    year = [2015,2016,2017,2018]
    my_labels={"1":'profit of 2015',"2":'profit of 2016',"3":'profit of 2017',"4":'profit of 2018'}
    plt.figure(figsize=(20, 10))
    for y in range(len(year)) :
        datelist = [f.split('_')[0] for f in os.listdir("./newstdcompare"+str(year[y])+"/")]
        actions_profit = [0]*25
        for date in datelist :
            daily_actions = pd.read_csv(path_to_all_gt+str(date[0:4])+'/'+date+ext_of_groundtruth)
            for i in range(25):
                actions_profit[i] += daily_actions['action'+str(i+1)].sum()
        print(actions_profit)
        #print(actions[0,0])
        x = [action[0] for action in actions]
        plt.scatter(x,actions_profit,label = my_labels[str(y+1)])

    plt.xlabel("actions")
    plt.ylabel("profit")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
        
    
    
if __name__=='__main__':
    datelist = [f.split('_')[0] for f in os.listdir(path_to_compare)]
    print(datelist)
    #check_cost_gate(datelist)a
    profit_distribution(datelist)