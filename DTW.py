# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:15:31 2021

@author: allen
"""
import os 
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn import preprocessing
from dtw import *
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

path_to_average = "./2017/averageprice/"
ext_of_average = "_averagePrice_min.csv"
path_to_2017compare = "./newstdcompare2017/"
ext_of_compare = "_table.csv"
path_to_dtw = "./newstdcomparedtw2017/"
def Dynamic_Time_Warping(mindata ,stock1 ,stock2):
    stock1_series = mindata[str(stock1)].values
    stock1_series = preprocessing.scale(stock1_series)
    stock2_series = mindata[str(stock2)].values
    stock2_series = preprocessing.scale(stock2_series)
    #alignment = dtw(stock1_series, stock2_series, keep_internals=True)
    alignment = dtw(query, template, keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="twoway",offset=-2)
    
    return

def build_dynamic_time_warping_index(index1s, index2s):
    matrix = np.zeros((150, 150),dtype=np.int)
    #print(matrix)
    for i in range(len(index1s)) :
            matrix[index1s[i],index2s[i]] = 1
    #print(matrix)
    return matrix
def Dynamic_Time_Warping_with_cointegration(mindata ,pair , table):
    stock1_series = mindata[str(table.stock1[pair])].values
    stock2_series = mindata[str(table.stock2[pair])].values
    #print("stock2_old_series",stock2_series)
    new_stock1_series = table.w1[pair] * np.log(stock1_series)
    new_stock2_series = -table.w2[pair] * np.log(stock2_series)+table.mu[pair]
    #print(stock1_series)
   # print(stock2_series)
    #spread = table.w1[pair] * np.log(tick_data[str(table.stock1[pair])]) + table.w2[pair] * np.log(
        #tick_data[str(table.stock2[pair])])
    #alignment = dtw(stock1_series, stock2_series, keep_internals=True)
    #alignment.plot(xlab = str(table.stock1[pair]) , ylab = str(table.stock2[pair]) ,type="threeway")

    alignment = dtw(new_stock1_series, new_stock2_series, keep_internals=True, 
    window_type="sakoechiba", window_args={'window_size': 10})
    
    #alignment.plot(type="twoway",offset=0)
    alignment.plot(xlab = str(table.stock1[pair]) , ylab = str(table.stock2[pair]) ,type="threeway")
    #print(alignment.index1s)
    #print(alignment.index2s)
    matrix = build_dynamic_time_warping_index(alignment.index1s,alignment.index2s)
    dynamic_stock2_series = []
    for i in range(len(matrix)):
        #print(matrix[i,:])
        v = np.argwhere(matrix[i,:] == 1)
        v = v.flatten().tolist()
        #print(v)
        new_values = 0
        for j in v :
            new_values += stock2_series[j]
            #print(new_values)
        new_values = new_values / len(v)
        dynamic_stock2_series.append(new_values)
    #print(dynamic_stock2_series-stock2_series)
    print(len(dynamic_stock2_series))
    new_dynamic_stock2_series = -table.w2[pair] * np.log(dynamic_stock2_series)+table.mu[pair]
    
    """
    recaculate the weight of cointegration 
    write model selection function
    """
    alignment2 = dtw(new_stock1_series, new_dynamic_stock2_series, keep_internals=True, window_type="sakoechiba", window_args={'window_size': 10})
    #alignment2.plot(type="twoway",offset=0)
    print(alignment.distance-alignment2.distance)
    """
    spread = table.w1[pair] * np.log(stock1_series) + table.w2[pair] * np.log(dynamic_stock2_series)
    plt.figure()
    plt.plot(spread)
    plt.show()
    """  
    
    #alignment.plot(xlab = stock1 , ylab = stock2 ,type="threeway")

    return alignment.distance
def main():
    choose = 1
    datelist = [f.split('_')[0] for f in os.listdir(path_to_average)]
    #print(datelist)
    count = 0
    for date in datelist: 
        if date == "20170103":
            mindata = pd.read_csv(path_to_average+date+ext_of_average)
            table = pd.read_csv(path_to_2017compare+date+ext_of_compare)
            mindata = mindata.iloc[16:166]
            #print(mindata)
            #print(mindata.columns)
            all_combinations = combinations(mindata.columns,2)
            if choose == 0 :
                for i in all_combinations: 
                    if count <=10 :
                        Dynamic_Time_Warping(mindata,i[0],i[1])
                    count += 1
            else :
                    num = np.arange(0,len(table),1)    
                    dtw_distance = []                     
                    for pair in num:
                        dis = Dynamic_Time_Warping_with_cointegration(mindata ,pair, table)
                        count += 1
                            #dtw_distance.append(dis)
                    #table['dtw_distance'] = dtw_distance
                    #table.to_csv(path_to_dtw+str(date)+'_table.csv', mode='w',index=False)

if __name__ == "__main__":
    main()