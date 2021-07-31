# -*- coding: utf-8 -*-
"""
Created on Tue May 26 20:06:37 2020

@author: allen
"""
import numpy as np
import pandas as pd
import os 
import collections
from sklearn import preprocessing
import matplotlib.pyplot as plt
#from sklearn.datasets import make_moons
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler 
ext_of_profit = "_profit.csv"
path_to_profit ='C:/Users/Allen/pair_trading DL2/profit pairs/'

if __name__ =="__main__":

    datelist = [f.split('_')[0] for f in os.listdir(path_to_profit)]

    reward=[]
    cumulative_reward=[]
    for i,date in enumerate(sorted(datelist)):
        #print(i,date)
        profit = pd.read_csv(path_to_profit+date+ext_of_profit)
        #print(profit)
        reward.append(profit["reward"].sum())
        total_reward = 0
    for i in range(len(reward)):
        temp = 0
        for j in range(i+1):
            temp+=reward[j]
        cumulative_reward.append(temp)
    print(cumulative_reward)
            
    maxi = 30000000       
    #R = pd.DataFrame(cumulative_reward)
    R = pd.DataFrame(x+maxi for x in cumulative_reward)
    print(R)
    r = (R - R.shift(1))/R.shift(1)
    print(r)
    sr = r.mean()/r.std() * np.sqrt(252)
    print(sr)
