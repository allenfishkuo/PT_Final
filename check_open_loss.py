# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:12:31 2020

@author: Allen
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:13:25 2020

@author: Allen
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
path_to_all = "./gt_all_new/"

from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
min_max_scaler = preprocessing.MinMaxScaler()

SS = StandardScaler()
def check_distributed():

    train_label = []
    test_label = []
    for year in range(2013,2017):
        path_to_year = path_to_all+str(year)+'/'
        gt_date = [f.split('_')[0] for f in os.listdir(path_to_year)]
       # print(gt_date)
        count2 = 0
        for date in sorted(gt_date):
            count2 +=1
            gt = pd.read_csv(path_to_year+date+ext_of_groundtruth,usecols=["open","loss"])
            #print(gt.iloc[0,0])
            gt = gt.values
            #print(gt)
            for l in range(len(gt)):
                    train_label.append(gt[l])
        
    train_label = np.asarray(train_label)
   # print(train_label.shape)
    #plt.figure()
    unique, counts = np.unique(train_label,axis = 0, return_counts=True)
   # print(type(counts))
    #counts = np.reshape(counts,(1,len(counts)))
   # print(counts)
    
    my_labels={"1":'prob : # of Trades:> 1 %',"2":'prob # of Trades:> 0.5% ',"3":'prob # of Trades: > 0.1%',"4":'Kmeans centers',"5":'prob # of Trades: < 0.1%',"6":"top_frequency"}
    exactly_label = []
    for i in range(len(unique)):
        if counts[i] >= 3050:
            exactly_label.append([unique[i,0],unique[i,1]])
   # print("# >1000 label :",exactly_label)
   # print(len(exactly_label))
    # 刪除低於目標個數的
    
    less_label = []
    t = 0
    """
    for i in range(len(unique)):
        if counts[i-t] <= 500:        
            less_label.append([unique[i-t,0],unique[i-t,1]])
            unique = np.delete(unique,i-t,axis = 0)
            counts = np.delete(counts,i-t)
            t += 1
    """        
    true_train_label = []
    for [x,y] in train_label.tolist():
        if [x,y] not in less_label :
            true_train_label.append([x,y])
    true_train_label = np.asarray(true_train_label)
    
    plt.figure()

    for i in range(len(unique)):
        if counts[i] >= 1050 :
            plt.scatter(unique[i,0],unique[i,1], c ="orange",alpha=0.9)
        elif counts[i] >= 1000 :
            plt.scatter(unique[i,0],unique[i,1], c ="red",alpha=0.4)
            
        elif counts[i] < 1000 and counts[i] >= 500:
            plt.scatter(unique[i,0],unique[i,1], c ="yellow",alpha=0.4)
            
        elif counts[i] < 500 and counts[i] > 100:
            plt.scatter(unique[i,0],unique[i,1], c ="green",alpha=0.4)
        else :
            plt.scatter(unique[i,0],unique[i,1], c ="blue",alpha=0.4)
    
           
 
    
            
    
    #print(counts)
    #model = SpectralClustering(n_clusters=4, affinity='nearest_neighbors',assign_labels='kmeans')
    #labels = model.fit_predict(train_label)
    #plt.scatter(train_label[:, 0], train_label[:, 1], c=labels,s=50, cmap='viridis')
    thresholds =[]
    for cluster in range(2,35,2) :
        kmeans = KMeans(n_clusters=cluster)
        kmeans.fit(train_label)
        centers = kmeans.cluster_centers_ 
        ratios = kmeans.inertia_
        y_kmeans = kmeans.predict(true_train_label)
        #print(ratios)
        
        plt.scatter(unique[0,0],unique[0,1], c ="red",alpha=0.4,label=my_labels["1"])
        plt.scatter(unique[0,0],unique[0,1], c ="yellow",alpha=0.4,label=my_labels["2"])
        plt.scatter(unique[0,0],unique[0,1], c ="green",alpha=0.4,label=my_labels["3"])
        plt.scatter(unique[0,0],unique[0,1], c ="blue",alpha=0.4,label=my_labels["5"])
        #plt.scatter(centers[:, 0], centers[:, 1], c='black',alpha=0.7,label=my_labels["4"])
        plt.scatter(unique[0,0],unique[0,1], c ="orange",alpha=0.9,label=my_labels["6"])
        means=[]
        #print(y_kmeans)
        #for i in range(len(train_label)):
        #plt.scatter(train_label[:,0],train_label[:,1], c =y_kmeans,cmap='viridis')
        for i in range(len(centers)):
            means.append([centers[i,0],centers[i,1]])
        
        #plt.title("Ground truth distributed")
        """
        ax.set_xlabel(Opening Trigge)
        ax.set_ylabel("Stop-Loss Trigger")
        """
        plt.xlabel("Opening Trigger")
        plt.ylabel("Stop loss Trigger")
        #ax.set_zlabel("# of optimal point")
        plt.legend(loc='lower right')
        plt.tight_layout()
        #plt.savefig(path_to_image+"Pairs trading Kmeans.png")
        plt.show()
        plt.close()
        
        print("cluster of "+str(cluster)+":",cluster,sorted(means))
        plt.figure()
        thresholds.append(sorted(means))
    return thresholds
    #visualizer = KElbowVisualizer(kmeans, k=(15,40))
    
    #visualizer.fit(train_label)        # Fit the data to the visualizer
    #visualizer.show()
    
    
        
        
        
check_distributed()