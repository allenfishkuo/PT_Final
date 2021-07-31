# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 12:53:20 2021

@author: allen
"""

import numpy as np
import pandas as pd
import os 
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
path_to_2013avg = "./2013/averageprice/"
path_to_2014avg = "./2014/averageprice/"
path_to_2015avg = "./2015/averageprice/"
path_to_2016avg = "./2016/averageprice/"
path_to_2017avg = "./2017/averageprice/"
path_to_2018avg = "./2018/averageprice/"
ext_of_average = "_averagePrice_min.csv"

path_to_minprice = "C:/Users/Allen/pair_trading DL2/2016/minprice/"
ext_of_minprice = "_min_stock.csv"

#path_to_half = "C:/Users/Allen/pair_trading DL/2016/2016_halfmin/"   #2016halfmin data
path_to_2013half = "./2013_halfmin/"
path_to_2014half = "./2014_halfmin/"
path_to_2015half = "./2015_halfmin/"
path_to_2016half = "./2016_halfmin/"
path_to_2017half = "./2017_halfmin/"
path_to_2018half = "./2018_halfmin/"
ext_of_half = "_half_min.csv"

#path_to_compare = "C:/Users/Allen/pair_trading DL/compare/"      #2016 halfmin spread w1w2

ext_of_compare = "_table.csv"
path_to_2013compare = "./newstdcompare2013/" 
path_to_2014compare = "./newstdcompare2014/" 
path_to_2015compare = "./newstdcompare2015/" 
path_to_2016compare = "./newstdcompare2016/" 
path_to_2017compare = "./newstdcompare2017/" #"./new_cluster_2017/5min_moving/"
#path_to_2017compare = "./new_cluster_2017/5min/"

path_to_2018compare = "./newstdcompare2018/" 
path_to_python ="C:/Users/Allen/pair_trading DL2"
path_to_groundtruth = "C:/Users/Allen/pair_trading DL2/ground truth trading period/"
ext_of_groundtruth = "_ground truth.csv"

gt_location = "gt_25action_0121"
path_to_choose2013 = "./gt_25action_0121/2013/"
path_to_choose2014 = "./gt_25action_0121(Method3)/2014/"
path_to_choose2015 = "./gt_25action_0121(Method3)/2015/"
path_to_choose2016 = "./gt_25action_0121(Method3)/2016/"
path_to_choose2017 = "./gt_25action_0121(Method3)/2017/"
path_to_choose2018 = "./gt_25action_new/2018/"


path_to_actions = "./gt_25action_0121/"
min_max_scaler = preprocessing.MinMaxScaler()

#no_half =["2231","8454","6285","2313","2867","1702","3662","1536","9938","2847","6456"]
min_max_scaler = preprocessing.MinMaxScaler()

SS = StandardScaler()
read_coverge_time = False
normalize_spread = False
input_of_three = False
Use_avg = False
def read_data():
    number_of_kmean = 25
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    count_number =[0]*number_of_kmean
    count_test =[0]*number_of_kmean

    dic_avg = {0 :path_to_2013avg, 1 :path_to_2014avg, 2 :path_to_2015avg,3:path_to_2016avg, 4:path_to_2017avg, 5:path_to_2018avg}
    dic_compare = { 0 : path_to_2013compare, 1 : path_to_2014compare, 2 : path_to_2015compare ,3:path_to_2016compare , 4:path_to_2017compare ,5:path_to_2018compare}
    dic_half = { 0 :path_to_2013half, 1 :path_to_2014half, 2 :path_to_2015half,3:path_to_2016half, 4:path_to_2017half, 5:path_to_2018half}
    dic_choose = { 0 : path_to_choose2013, 1 :path_to_choose2014, 2 :path_to_choose2015,3:path_to_choose2016, 4:path_to_choose2017, 5 :path_to_choose2018}
    for year in range(3,len(dic_choose)-2):
        print(dic_compare[year])
        datelist = [f.split('_')[0] for f in os.listdir(dic_compare[year])] #選擇幾年度的
        #print(dic_compare[year])
        #print(datelist)
        count = 0
        for date in sorted(datelist):
            #if date[0:6] == "201501" or date[0:6] == "201502" or date[0:6] == "201503" or date[0:6] == "201504" or date[0:6] == "201505" or date[0:6] == "201506" :
            if date[0:6] == "201601" or date[0:6] == "201602" or date[0:6] == "201603" or date[0:6] == "201604" or date[0:6] == "201605" or date[0:6] == "201606" :
                continue
            #print(date)
            count +=1
            table = pd.read_csv(dic_compare[year]+date+ext_of_compare)
            if Use_avg : avgmin = pd.read_csv(dic_avg[year]+date+ext_of_average)
            else : halfmin = pd.read_csv(dic_half[year]+date+ext_of_half)
            gt = pd.read_csv(dic_choose[year]+date+ext_of_groundtruth,usecols=["action choose"])
            gt = gt.values
            #print(date)
            #print(count)

            for pair in range(len(table)):  
                if Use_avg :      
                    spread = table.w1[pair] * np.log((avgmin[ str(table.stock1[pair]) ])) + table.w2[pair] * np.log(avgmin[ str(table.stock2[pair]) ])
                    spread = spread[16:166].values
                    #spread = preprocessing.scale(spread)
                    if normalize_spread :
                        spread[:] = (spread[:] - table.mu[pair])/table.stdev[pair]
                        new_spread = np.zeros((1,512))
                        new_spread[0,181:331] = spread 
                    else :            
                        
                        spread[:] = (spread[:] - table.mu[pair])/table.stdev[pair]
                        new_spread = np.zeros((3,512))
                        new_spread[0,181:331] = spread    
                        mindata1 = avgmin[str(table.stock1[pair])][16:166].values
                        mindata2 = avgmin[str(table.stock2[pair])][16:166].values
                        mindata1 = preprocessing.scale(mindata1)
                        mindata2 = preprocessing.scale(mindata2)       
                        new_spread[1,181:331] = mindata1
                        new_spread[2,181:331] = mindata2
                else :
                    spread = table.w1[pair] * np.log((halfmin[ str(table.stock1[pair]) ])) + table.w2[pair] * np.log(halfmin[ str(table.stock2[pair]) ])
                    spread = spread[32:332].values
                    #spread = preprocessing.scale(spread)
                    if normalize_spread == True :
                        spread[:] = (spread[:] - table.mu[pair])/table.stdev[pair]
                        new_spread = np.zeros((1,512))
                        new_spread[0,106:406] = spread 
                    else :                        
                        spread[:] = (spread[:] - table.mu[pair])/table.stdev[pair]
                        new_spread = np.zeros((3,512))
                        new_spread[0,106:406] = spread    
                        mindata1 = (halfmin[str(table.stock1[pair])][33:333].values-halfmin[str(table.stock1[pair])][32:332].values)/halfmin[str(table.stock1[pair])][32:332].values
                        mindata2 = (halfmin[str(table.stock2[pair])][33:333].values-halfmin[str(table.stock2[pair])][32:332].values)/halfmin[str(table.stock2[pair])][32:332].values
                        mindata1 = preprocessing.scale(mindata1)
                        mindata2 = preprocessing.scale(mindata2)       
                        new_spread[1,106:406] = mindata1
                        new_spread[2,106:406] = mindata2
                        """
                        plt.figure()
                        for i in range(new_spread.shape[0]):
                            plt.plot(new_spread[i,:])
                        plt.show()
                        plt.close()
                        """
                if date[0:6] == "201611" or date[0:6] =="201612" : #validation data
                            number = gt[pair][0]
                            for i in range(number_of_kmean): #幾個action
                                if number == i  :
                                    test_data.append(new_spread)
                                    count_test[number] +=1
                                    test_label.append(gt[pair])
                else :            
                    #if date[0:6] != "201501" and date[0:6] !="201502" and date[0:6] !="201503" and date[0:6] !="201504" and  date[0:6] !="201505" and date[0:6] !="201506" :
                        number = gt[pair][0]
                        for i in range(number_of_kmean): #幾個action
                            if number == i : 
                                train_data.append(new_spread)
                                count_number[number] +=1
                                train_label.append(gt[pair])                                   
    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)
    
    
    #print(train_data)
    #train_data = preprocessing.scale(train_data,axis=1)
    #test_data = preprocessing.scale(test_data,axis=1)
    #train_data = preprocessing.normalize(train_data,norm ="l1")
    """
    for i in range(train_data.shape[1]):
            train_data[:, i, :] = preprocessing.scale(train_data[:, i, :],axis=1)
    print(train_data)
    
    for i in range(test_data.shape[1]):
            test_data[:, i, :] = preprocessing.scale(test_data[:, i, :],axis=1)
    #test_data = preprocessing.scale(test_data,axis=1)
    #test_data = preprocessing.normalize(test_data,norm ="l1")
    print(train_data.shape)
    """
    #print(train_data)
  
    train_label = np.asarray(train_label)
    test_label = np.asarray(test_label)
    train_label = train_label.flatten()
    test_label = test_label.flatten()
    print(train_label.shape)
    print(count_number)
    print(count_test)
    print(train_data.shape)
    
    print(test_data.shape)
    
    return train_data, train_label, test_data, test_label
path_to_threshold = "./model/2013-2014_amsgrad_0120(M3)/threshold_label(ST)/"
def find_threshold_data():
    path_to_minprice = "./2015/minprice/"
    trading_cost_threshold = np.arange(0.0025,0.008,0.0005)
    number_of_label = len(trading_cost_threshold)
    count_trading_cost_threshold = [0] * number_of_label
    find_threshold_data = []
    #counts = [0,0,0,0,0]
    datelist = [f.split('_')[0] for f in os.listdir(path_to_2015compare)]
    month_list =[]
    #print(datelist)
    count = 0
    for date in sorted(datelist[:]): 
       # if date[:6] == '201611' or date[:6] == '201612':
            count +=1
            table = pd.read_csv(path_to_2015compare+date+ext_of_compare)
            if Use_avg : avgmin = pd.read_csv(path_to_2015avg+date+ext_of_average)
            else : halfmin = pd.read_csv(path_to_2015half+date+ext_of_half)            
            try:
                tickdata = pd.read_csv(path_to_minprice+date+ext_of_minprice).drop([266, 267, 268, 269, 270])
            except:
                continue
            for pair in range(len(table)):
                if Use_avg :
                    spread = table.w1[pair] * np.log((avgmin[ str(table.stock1[pair]) ])) + table.w2[pair] * np.log(avgmin[ str(table.stock2[pair]) ])
                    spread = spread[16:166].values
                    spread[:] = (spread[:] - table.mu[pair])/table.stdev[pair]
                    new_spread = np.zeros((3,512))
                    new_spread[0,181:331] = spread    
                    mindata1 = avgmin[str(table.stock1[pair])][16:166].values
                    mindata2 = avgmin[str(table.stock2[pair])][16:166].values
                    mindata1 = preprocessing.scale(mindata1)
                    mindata2 = preprocessing.scale(mindata2)       
                    new_spread[1,181:331] = mindata1
                    new_spread[2,181:331] = mindata2
                    find_threshold_data.append(new_spread )
                else :
                    spread = table.w1[pair] * np.log((halfmin[ str(table.stock1[pair]) ])) + table.w2[pair] * np.log(halfmin[ str(table.stock2[pair]) ])
                    spread = spread[32:332].values
                    spread[:] = (spread[:] - table.mu[pair])/table.stdev[pair]                    
                    new_spread = np.zeros((3,512))
                    new_spread[0,106:406] = spread                        
                    mindata1 = halfmin[str(table.stock1[pair])][32:332].values
                    mindata2 = halfmin[str(table.stock2[pair])][32:332].values
                    mindata1 = preprocessing.scale(mindata1)
                    mindata2 = preprocessing.scale(mindata2)
                    new_spread[1,106:406] = mindata1
                    new_spread[2,106:406] = mindata2                    
                    find_threshold_data.append(new_spread )
        
    find_threshold_data = np.asarray(find_threshold_data)
    print("whole_year :",find_threshold_data.shape)

    return find_threshold_data
def val_data():
    path_to_minprice = "./2015/minprice/"
    number_of_label = 2
    count_trading_cost_threshold = [0] * number_of_label
    threshold_data = []
    threshold_label = []
    #counts = [0,0,0,0,0]
    datelist = [f.split('_')[0] for f in os.listdir(path_to_2015compare)]
    month_list =[]
    #print(datelist)
    count = 0
    for date in sorted(datelist[:]): 
       # if date[:6] == '201611' or date[:6] == '201612':
            count +=1
            table = pd.read_csv(path_to_2015compare+date+ext_of_compare)
            if Use_avg : avgmin = pd.read_csv(path_to_2015avg+date+ext_of_average)
            else : halfmin = pd.read_csv(path_to_2015half+date+ext_of_half)       
            try:
                tickdata = pd.read_csv(path_to_minprice+date+ext_of_minprice).drop([266, 267, 268, 269, 270])
            except:
                continue
            gt = pd.read_csv(path_to_threshold+date+ext_of_groundtruth,usecols=["theshold choose"])
            gt = gt.values
            
            for pair in range(len(table)):
                if Use_avg :
                    spread = table.w1[pair] * np.log((avgmin[ str(table.stock1[pair]) ])) + table.w2[pair] * np.log(avgmin[ str(table.stock2[pair]) ])
                    spread = spread[16:166].values
                    spread[:] = (spread[:] - table.mu[pair])/table.stdev[pair]
                    new_spread = np.zeros((3,512))
                    new_spread[0,181:331] = spread    
                    mindata1 = avgmin[str(table.stock1[pair])][16:166].values
                    mindata2 = avgmin[str(table.stock2[pair])][16:166].values
                    mindata1 = preprocessing.scale(mindata1)
                    mindata2 = preprocessing.scale(mindata2)       
                    new_spread[1,181:331] = mindata1
                    new_spread[2,181:331] = mindata2
                    number = gt[pair][0]

                    for i in range(number_of_label): #幾個action
                        if number == i : 
                            threshold_data.append(new_spread)
                            count_trading_cost_threshold[number] +=1
                            threshold_label.append(gt[pair])
                else :
                    spread = table.w1[pair] * np.log((halfmin[ str(table.stock1[pair]) ])) + table.w2[pair] * np.log(halfmin[ str(table.stock2[pair]) ])
                    spread = spread[32:332].values
                    spread[:] = (spread[:] - table.mu[pair])/table.stdev[pair]
                    
                    new_spread = np.zeros((3,512))
                    new_spread[0,106:406] = spread    
                    
                    mindata1 = halfmin[str(table.stock1[pair])][32:332].values
                    mindata2 = halfmin[str(table.stock2[pair])][32:332].values
                    mindata1 = preprocessing.scale(mindata1)
                    mindata2 = preprocessing.scale(mindata2)
                    new_spread[1,106:406] = mindata1
                    new_spread[2,106:406] = mindata2
                    number = gt[pair][0]
                    for i in range(number_of_label): #幾個action
                        if number == i : 
                            threshold_data.append(new_spread)
                            count_trading_cost_threshold[number] +=1
                            threshold_label.append(gt[pair])            
    threshold_data = np.asarray(threshold_data)
    threshold_label = np.asarray(threshold_label)
    threshold_label = threshold_label.flatten()
    print(threshold_label.shape)
    print(count_trading_cost_threshold)
    print(threshold_data.shape)

    return threshold_data, threshold_label
def test_data():
    whole_year = []
    #test_data = []
    path_to_minprice = "./2017/minprice/"
    #test_label = []
    #counts = [0,0,0,0,0]
    datelist = [f.split('_')[0] for f in os.listdir(path_to_2017compare)]
    month_list =[]
    #print(datelist)
    count = 0
    #same_date = ['201701','201702','201703','201704','201705','201706','201707','201708','201709','201710','201711','201712']   
    for date in sorted(datelist[:]): 
            count +=1
            table = pd.read_csv(path_to_2017compare+date+ext_of_compare)
            if Use_avg : avgmin = pd.read_csv(path_to_2017avg+date+ext_of_average)
            else : halfmin = pd.read_csv(path_to_2017half+date+ext_of_half)
            
            try:
                tickdata = pd.read_csv(path_to_minprice+date+ext_of_minprice).drop([266, 267, 268, 269, 270])
            except:
                continue            
            
            for pair in range(len(table)):
                if Use_avg :
                    spread = table.w1[pair] * np.log((avgmin[ str(table.stock1[pair]) ])) + table.w2[pair] * np.log(avgmin[ str(table.stock2[pair]) ])
                    spread = spread[16:166].values
                    if normalize_spread :
                        spread[:] = (spread[:] - table.mu[pair])/table.stdev[pair]
                        new_spread = np.zeros((1,512))
                        new_spread[0,181:331] = spread 
                    else :                        
                        spread[:] = (spread[:] - table.mu[pair])/table.stdev[pair]
                        new_spread = np.zeros((3,512))
                        new_spread[0,181:331] = spread    
                        mindata1 = avgmin[str(table.stock1[pair])][16:166].values
                        mindata2 = avgmin[str(table.stock2[pair])][16:166].values
                        mindata1 = preprocessing.scale(mindata1)
                        mindata2 = preprocessing.scale(mindata2)       
                        new_spread[1,181:331] = mindata1
                        new_spread[2,181:331] = mindata2
                else :
                    spread = table.w1[pair] * np.log((halfmin[ str(table.stock1[pair]) ])) + table.w2[pair] * np.log(halfmin[ str(table.stock2[pair]) ])
                    spread = spread[32:332].values
                    #spread = preprocessing.scale(spread)
                    if normalize_spread  :
                        spread[:] = (spread[:] - table.mu[pair])/table.stdev[pair]
                        new_spread = np.zeros((1,512))
                        new_spread[0,106:406] = spread 
                    else :                        
                        spread[:] = (spread[:] - table.mu[pair])/table.stdev[pair]
                        new_spread = np.zeros((3,512))
                        new_spread[0,106:406] = spread    
                        mindata1 = (halfmin[str(table.stock1[pair])][33:333].values-halfmin[str(table.stock1[pair])][32:332].values)/halfmin[str(table.stock1[pair])][32:332].values
                        mindata2 = (halfmin[str(table.stock2[pair])][33:333].values-halfmin[str(table.stock2[pair])][32:332].values)/halfmin[str(table.stock2[pair])][32:332].values
                        mindata1 = preprocessing.scale(mindata1)
                        mindata2 = preprocessing.scale(mindata2)       
                        new_spread[1,106:406] = mindata1
                        new_spread[2,106:406] = mindata2
                    
                whole_year.append(new_spread )
                    
            
    print(month_list)            
    whole_year = np.asarray(whole_year)
    print("whole_year :",whole_year.shape)

    return whole_year
import json
import ast
def read_actions():
    f = open(path_to_actions+'open_loss.txt')
    text = f.read()
    print(text)
    x = ast.literal_eval(text)
    print(type(x))
    f.close
if __name__ == '__main__':    
    read_data()
    #test_data()
    #val_data()
    #find_threshold_data()
    #read_actions()

