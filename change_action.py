# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 11:20:39 2021

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
from mpl_toolkits import mplot3d
from matplotlib import cm
path_to_average = "C:/Users/Allen/pair_trading DL/2016/averageprice/"
ext_of_average = "_averagePrice_min.csv"
path_to_minprice = "C:/Users/Allen/pair_trading DL/2016/minprice/"
ext_of_minprice = "_min_stock.csv"
path_to_compare = "C:/Users/Allen/pair_trading DL/compare/"
ext_of_compare = "_table.csv"
path_to_python ="C:/Users/Allen/pair_trading DL"
path_to_groundtruth = "C:/Users/Allen/pair_trading DL/ground truth trading period/"
ext_of_groundtruth = "_ground truth.csv"
path_to_choose = "C:/Users/Allen/pair_trading DL/action_choose/"
path_to_all = "./gt_all/"
path_to_all_new = "./gt_all_new/"

def change_action():
    for year in range(2013,2015):
        path_to_year = path_to_all+str(year)+'/'
        gt_date = [f.split('_')[0] for f in os.listdir(path_to_year)]
       # print(gt_date)
        count2 = 0
        for date in sorted(gt_date):
            count2 +=1
            gt = pd.read_csv(path_to_year+date+ext_of_groundtruth)
            for i in range(len(gt)):
                print(gt.loc[i,'reward'])
                if gt.loc[i,'reward'] <= 0 :
                    gt.loc[i,'open'] = 10
                    gt.loc[i,'loss'] = 25
            gt.to_csv(path_to_all_new+str(date[0:4])+'/'+str(date)+'_ground truth.csv', mode='w',index=False)
if __name__ == "__main__":
    change_action()