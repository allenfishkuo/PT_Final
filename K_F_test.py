# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 12:04:52 2021

@author: allen
"""

import numpy as np
import pandas as pd
import os 
from sklearn import preprocessing
import matplotlib.pyplot as pltr
from sklearn.preprocessing import StandardScaler 
import random

import matplotlib.mlab as mlab

from mpmath import norm

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
ext_of_groundtruth = "_ground truth.csv"

ext_of_compare = "_table.csv"
path_to_2013compare = "./newstdcompare2013/" 
path_to_2014compare = "./newstdcompare2014/" 
path_to_2015compare = "./newstdcompare2015/" 
path_to_2016compare = "./newstdcompare2016/" 
path_to_2017compare = "./newstdcompare2017/" #"./new_cluster_2017/5min_moving/"
#path_to_2017compare = "./new_cluster_2017/5min/"
path_to_2018compare = "./newstdcomparedtw2018/" 


path_to_2018compare = "./newstdcompare2018/" 
dic_avg = {0 :path_to_2013avg, 1 :path_to_2014avg, 2 :path_to_2015avg,3:path_to_2016avg, 4:path_to_2017avg, 5:path_to_2018avg}
dic_compare = { 0 : path_to_2013compare, 1 : path_to_2014compare, 2 : path_to_2015compare ,3:path_to_2016compare , 4:path_to_2017compare ,5:path_to_2018compare}
dic_half = { 0 :path_to_2013half, 1 :path_to_2014half, 2 :path_to_2015half,3:path_to_2016half, 4:path_to_2017half, 5:path_to_2018half}
#dic_choose = { 0 : path_to_choose2013, 1 :path_to_choose2014, 2 :path_to_choose2015,3:path_to_choose2016, 4:path_to_choose2017, 5 :path_to_choose2018}

def get_group(g, key):
   if key in g.groups: return g.get_group(key)
   return pd.DataFrame()


def change_df_to_np(data):
    new_data = data.values
    new_data = new_data.flatten()
    new_data = np.sort(new_data)
    return new_data
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.stats import norm
data_train = pd.DataFrame()
data_train2 = pd.DataFrame()

data_test = pd.DataFrame()
data_test2 = pd.DataFrame()

sp_train = []
sp_train_std = []
sp_test_std = []
return_train_std = []
return_train = []
return_test = []
return_test_std = []
sp_test = []
count1 ,count2 = 0,0

#loc, scale = norm.fit(data)
for year in range(3,len(dic_compare)-1):
        print(dic_compare[year])
        datelist = [f.split('_')[0] for f in os.listdir(dic_compare[year])] #選擇幾年度的
        #print(dic_compare[year])
        #print(datelist)
        
        count = 0
        for date in sorted(datelist):
            #if date[:6] != "201501":
             #   continue

            count +=1
            table = pd.read_csv(dic_compare[year]+date+ext_of_compare)
            avgmin = pd.read_csv(dic_avg[year]+date+ext_of_average)

            gt = pd.read_csv(dic_compare[year]+date+ext_of_compare,usecols=["Emu","Estd","model_type"])

            if date[:6] =="201611" :
                data_train = pd.concat([data_train,gt["Emu"]],ignore_index = True )
                data_train2 = pd.concat([data_train2,gt["Estd"]],ignore_index = True )
            if  date[:6] == "201612" :
                data_test = pd.concat([data_test,gt["Emu"]],ignore_index = True )
                data_test2 = pd.concat([data_test2,gt["Estd"]],ignore_index = True )


            for pair in range(len(table)):
                mindata1 = avgmin[str(table.stock1[pair])][16:166].mean()
                mindata2 = avgmin[str(table.stock2[pair])][16:166].mean()
                mindata3 = avgmin[str(table.stock1[pair])][16:166].std()
                mindata4 = avgmin[str(table.stock2[pair])][16:166].std()
                return1 = ((avgmin[str(table.stock1[pair])][17:167].values-avgmin[str(table.stock1[pair])][16:166].values)/avgmin[str(table.stock1[pair])][16:166].values).mean()   
                return2 = ((avgmin[str(table.stock2[pair])][17:167].values-avgmin[str(table.stock2[pair])][16:166].values)/avgmin[str(table.stock2[pair])][16:166].values).mean()
                return3 = ((avgmin[str(table.stock1[pair])][17:167].values-avgmin[str(table.stock1[pair])][16:166].values)/avgmin[str(table.stock1[pair])][16:166].values).std()   
                return4 = ((avgmin[str(table.stock2[pair])][17:167].values-avgmin[str(table.stock2[pair])][16:166].values)/avgmin[str(table.stock2[pair])][16:166].values).std()

                if date[:6] == "201611"   :
                    #print(mindata1)
                    sp_train.append(mindata1)
                    sp_train.append(mindata2)
                    sp_train_std.append(mindata3)
                    sp_train_std.append(mindata4)
                    return_train.append(return1)
                    return_train.append(return2)
                    return_train_std.append(return3)
                    return_train_std.append(return4)
                if  date[:6] == "201612" :
                    sp_test.append(mindata1)
                    sp_test.append(mindata2)
                    sp_test_std.append(mindata3)
                    sp_test_std.append(mindata4)
                    return_test.append(return1)
                    return_test.append(return2)
                    return_test_std.append(return3)
                    return_test_std.append(return4)
#maxabs = preprocessing.StandardScaler()
#data_train = maxabs.fit_transform(data_train)
#data_test = maxabs.fit_transform(data_test)
sp_test = np.array(sp_test)
sp_train = np.array(sp_train)
return_train =np.array(return_train)
return_test = np.array(return_test)
sp_test_std = np.array(sp_test_std)
sp_train_std = np.array(sp_train_std)   
return_train_std =np.array(return_train_std)
return_test_std = np.array(return_test_std)


print(np.sort(sp_train))
print(np.sort(return_train))


data_train = change_df_to_np(data_train)
data_test = change_df_to_np(data_test)
data_train2 = change_df_to_np(data_train2)
data_test2 = change_df_to_np(data_test2)
return_train = np.sort(return_train)
return_train_std = np.sort(return_train_std)
return_test = np.sort(return_test)
return_test_std = np.sort(return_test_std)
sp_train = np.sort(sp_train)
sp_train_std = np.sort(sp_train_std)
sp_test = np.sort(sp_test)
sp_test_std = np.sort(sp_test_std)
data_train = data_train[(data_train < 10) & (data_train > -10)]
data_test = data_test[(data_test < 10) & (data_test > -10)]
data_train2 = data_train2[(data_train2 < 1) & (data_train2 > -1)]
data_test2 = data_test2[(data_test2 < 1) & (data_test2 > -1)]




print(np.isnan(sp_train),np.isnan(sp_test),np.isnan(return_train),np.isnan(return_test))

print("r1" ,ks_2samp(data_train,data_test))
print("r2",ks_2samp(return_train,return_test))
print("r3",ks_2samp(sp_train,sp_test))


print("s1" ,ks_2samp(data_train2,data_test2))
print("s2",ks_2samp(return_train_std,return_test_std))
print("s3",ks_2samp(sp_train_std,sp_test_std))
#n, bins, patches = plt.hist( data_train,density=True,  
 #                                    bins=np.arange(data_train.min(), data_train.max()+0.01, 0.01), 
#                                     rwidth=0.5 )
#x = np.arange(new_train.min(), new_train.max()+0.2, 0.2)
#y=norm.pdf(bins,loc,scale)

#plt.plot(bins, y, 'r--')
y1 = np.arange(len(data_train)) / float(len(data_train))
plt.plot(data_train,y1,label='spread mean train')

y2 = np.arange(len(data_test)) / float(len(data_test))
plt.plot(data_test,y2,label='spread mean test')
plt.legend()

plt.show()

y1 = np.arange(len(return_train)) / float(len(return_train))
plt.plot(return_train,y1,label='return mean train')

y2 = np.arange(len(return_test)) / float(len(return_test))
plt.plot(return_test,y2,label='return mean test')
plt.legend()

plt.show()

y1 = np.arange(len(sp_train)) / float(len(sp_train))
plt.plot(sp_train,y1,label='price mean train')

y2 = np.arange(len(sp_test)) / float(len(sp_test))
plt.plot(sp_test,y2,label='price mean test')
plt.legend()

plt.show()
#plt.plot(x, 1100*n.pdf(x))
y1 = np.arange(len(data_train2)) / float(len(data_train2))
plt.plot(data_train2,y1,label='spread std train')

y2 = np.arange(len(data_test2)) / float(len(data_test2))
plt.plot(data_test2,y2,label='spread std test')
plt.legend()

plt.show()

y1 = np.arange(len(return_train_std)) / float(len(return_train_std))
plt.plot(return_train_std,y1,label='return std train')

y2 = np.arange(len(return_test_std)) / float(len(return_test_std))
plt.plot(return_test_std,y2,label='return std test')
plt.legend()

plt.show()

y1 = np.arange(len(sp_train_std)) / float(len(sp_train_std))
plt.plot(sp_train_std,y1,label='price std train')

y2 = np.arange(len(sp_test_std)) / float(len(sp_test_std))
plt.plot(sp_test_std,y2,label='price std test')
plt.legend()

plt.show()



