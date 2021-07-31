import accelerate_formation
import ADF
import time
import pandas as pd
import numpy as np

import os, sys

from datetime import datetime
from sqlalchemy import create_engine
import pymysql

from vecm import para_vecm
from Matrix_function import order_select

# from sklearn import preprocessing
# import torch
# from ResNet.model import MSResNet
from itertools import combinations
from sklearn import preprocessing
from dtw import *

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from MTSA import order_select , snr , zcr , JB_VECM #, chow_test 
from vecm import vecm , rank , eig , weigh
from VecmPvalue import vecm_pvalue

def cointegration_weight( stock1, stock2 ):
    
    # 開啟matlab引擎
    #eng=matlab.engine.start_matlab()  
    
    # 選擇適合的 VECM model，並且檢定 formation period 是否有結構性斷裂，並刪除該配對，其餘配對則回傳共整合係數。
    #rank = 1
    #t1 = int(len(min_price)*3/4)     # 一天的時間長度(偵測兩天中間是否有結構性斷裂)
    
    local_select_model = []
    local_weight = []
    local_name = []
    local_pval = []
            
    # stock1 = min_price.iloc[:,i]
    # stock2 = min_price.iloc[:,j]
        
    # stock1_name = min_price.columns.values[i]
    # stock2_name = min_price.columns.values[j]
    
    z = ( np.vstack( [stock1 , stock2] ).T )
    model = VAR(z)
    p = order_select(z,5)
    #p = int(model.select_order(5).bic)
        
    # VAR 至少需落後1期
    if p < 1:     
            
        return 0, 0
        
    # # portmanteau test
    # if model.fit(p).test_whiteness( nlags = 5 ).pvalue < 0.05:
            
    #     return 0, 0
        
    # # Normality test
    # if model.fit(p).test_normality().pvalue < 0.05:
            
    #     return 0, 0
    
    #r1 = eng.rank_jci( matlab.double(z.tolist()) , 'H2' , (p-1) ) 
    #r2 = eng.rank_jci( matlab.double(z.tolist()) , 'H1*' , (p-1))
    #r3 = eng.rank_jci( matlab.double(z.tolist()) , 'H1' , (p-1) )
    
    r1 = rank( pd.DataFrame(z) , 'H2' , p ) 
    r2 = rank( pd.DataFrame(z) , 'H1*' , p )
    r3 = rank( pd.DataFrame(z) , 'H1' , p )
    #r4 = rank( pd.DataFrame(z) , 'H*' , p )
        
    if r3 > 0:                      # 在 model 3 上有 rank
            
        if r2 > 0:                  # 在 model 2 上有 rank
                
            if r1 > 0:              # select model 1 and model 2 and model 3
                    
                #lambda_model2 = eng.eig_jci( matlab.double(z.tolist()) , 'H1*' , (p-1) , r2 )
                #lambda_model3 = eng.eig_jci( matlab.double(z.tolist()) , 'H1' , (p-1) , r2 )
            
                lambda_model2 = eig( pd.DataFrame(z) , 'H1*' , p , r2 )
                lambda_model3 = eig( pd.DataFrame(z) , 'H1' , p , r2 )

                test = np.log(lambda_model2/lambda_model3) * (len(stock1)-p)
                if test <= 0:
                    return 0,0
                    raise ValueError('test value error')
                if test > 3.8414:
                
                    #bp1 = chow_test( z , t1 , p , 'H1' , r3 )
                
                    #if bp1 == 0:               
                    
                    local_select_model.append('model3')
                    return weigh( pd.DataFrame(z) , 'H1' , p , r3 )
                    #weight.append( eng.coin_jci( matlab.double(z.tolist()) , 'H1' , (p-1) , r3 ) )
                    # local_weight.append( weigh( pd.DataFrame(z) , 'H1' , p , r3 ) )
                
                    # local_name.append([stock1_name,stock2_name])
                    
                    # local_pval.append( vecm_pvalue('model3', vecm( pd.DataFrame(z),'H1',p)[0][0] ) )
                
                else:
                        
                    #lambda_model1 = eng.eig_jci( matlab.double(z.tolist()) , 'H2' , (p-1) , r1 )
                
                    lambda_model1 = eig( pd.DataFrame(z) , 'H2' , p , r1 )
                
                    test = np.log(lambda_model1/lambda_model2) * (len(stock1)-p)
                            
                    if test > 3.8414:
                    
                        #bp1 = chow_test( z , t1 , p , 'H1*' , r2 )
                    
                        #if bp1 == 0:    
                        
                        # local_select_model.append('model2')
                    
                        #weight.append( eng.coin_jci( matlab.double(z.tolist()) , 'H1*' , (p-1) , r2 ) )
                        return weigh( pd.DataFrame(z) , 'H1*' , p , r2 )
                        # local_weight.append( weigh( pd.DataFrame(z) , 'H1*' , p , r2 ) )
                            
                        # local_name.append([stock1_name,stock2_name])
                        
                        # local_pval.append( vecm_pvalue('model2',vecm(pd.DataFrame(z),'H1*',p)[0][1] ) ) 
                            
                    else:
                        
                        #bp1 = chow_test( z , t1 , p , 'H2' , r1 ) 
                
                        #if bp1 == 0:      
                    
                        # local_select_model.append('model1')
                
                        #weight.append( eng.coin_jci( matlab.double(z.tolist()) , 'H2' , (p-1) , r1 ) )
                        return weigh( pd.DataFrame(z) , 'H2' , p , r1 )
                        # local_weight.append( weigh( pd.DataFrame(z) , 'H2' , p , r1 ) )
                
                        # local_name.append([stock1_name,stock2_name])
                        
                        # local_pval.append( vecm_pvalue('model1',vecm(pd.DataFrame(z),'H2',p)[0][0] ) )
                
            else:                   # select model 2 and model 3
                
                #lambda_model2 = eng.eig_jci( matlab.double(z.tolist()) , 'H1*' , (p-1) , r2 )
                #lambda_model3 = eng.eig_jci( matlab.double(z.tolist()) , 'H1' , (p-1) , r2 )
            
                lambda_model2 = eig( pd.DataFrame(z) , 'H1*' , p , r2 )
                lambda_model3 = eig( pd.DataFrame(z) , 'H1' , p , r2 )

                test = np.log(lambda_model2/lambda_model3) * (len(stock1)-p)
                
                if test <= 0:
                    return 0,0
                    raise ValueError('test value error')
                if test > 3.8414:
                        
                    #bp1 = chow_test( z , t1 , p , 'H1' , r3 )
                
                    #if bp1 == 0:               
                    
                    # local_select_model.append('model3')
                
                    #weight.append( eng.coin_jci( matlab.double(z.tolist()) , 'H1' , (p-1) , r3 ) )
                    return weigh( pd.DataFrame(z) , 'H1' , p , r3 )
                    # local_weight.append( weigh( pd.DataFrame(z) , 'H1' , p , r3 ) )
                
                    # local_name.append([stock1_name,stock2_name])
                    
                    # local_pval.append( vecm_pvalue('model3',vecm(pd.DataFrame(z),'H1',p)[0][0] ) )
                
                else:
                        
                    #bp1 = chow_test( z , t1 , p , 'H1*' , r2 ) 
                
                    #if bp1 == 0:      
                    
                    # local_select_model.append('model2')
                
                    #weight.append( eng.coin_jci( matlab.double(z.tolist()) , 'H1*' , (p-1) , r2 ) )
                    return weigh( pd.DataFrame(z) , 'H1*' , p , r2 )
                    # local_weight.append( weigh( pd.DataFrame(z) , 'H1*' , p , r2 ) )
                
                    # local_name.append([stock1_name,stock2_name])
                    
                    # local_pval.append( vecm_pvalue('model2',vecm(pd.DataFrame(z),'H1*',p)[0][1] ) )
                
        else :                     # 只在 model 3 上有rank
                
            
            #bp1 = chow_test( z , t1 , p , 'H1' , r3 ) 
            
            #if bp1 == 0:            
                
            # local_select_model.append('model3')
            
            #weight.append( eng.coin_jci( matlab.double(z.tolist()) , 'H1' , (p-1) , r3 ) )
            return weigh( pd.DataFrame(z) , 'H1' , p , r3 )
            # local_weight.append( weigh( pd.DataFrame(z) , 'H1' , p , r3 ) )
            
            # local_name.append([stock1_name,stock2_name])
            
            # local_pval.append( vecm_pvalue('model3',vecm(pd.DataFrame(z),'H1',p)[0][0] ) )
    
            
        
    return 0,0


def spread_mean(stock1,stock2,w1,w2,_model):
    model = ""
    if _model == 'model1':
        model = 'H2'
    elif _model == 'model2':
        model = 'H1*'
    elif _model == 'model3':
        model = 'H1'
    stock1 = stock1[:150]
    stock2 = stock2[:150]
    b1 = w1
    b2 = w2
    y = np.vstack( [stock1, stock2] ).T
    logy = np.log(y)
    lyc = logy.copy()
    p = order_select(logy,5)
    _,_,para = para_vecm(logy,model,p)
    logy = np.mat(logy)
    y_1 = np.mat(logy[p:])
    dy = np.mat(np.diff(logy,axis=0))
    for j in range(len(stock1)-p-1):
        if model == 'H1':
            if p!=1:
                delta = para[0] * para[1].T * y_1[j].T + para[2] * np.hstack([dy[j:(j+p-1)].flatten(),np.mat([1])]).T
            else:
                delta = para[0] * para[1].T * y_1[j].T + para[2] * np.mat([1])
        elif model == 'H1*':
            if p!=1:
                delta = para[0] * para[1].T * np.hstack([y_1[j],np.mat([1])]).T + para[2] * dy[j:(j+p-1)].flatten().T
            else:
                delta = para[0] * para[1].T * np.hstack([y_1[j],np.mat([1])]).T
        elif model == 'H2':
            if p!=1:
                delta = para[0] * para[1].T * y_1[j].T + para[2] * dy[j:(j+p-1)].flatten().T
            else:
                delta = para[0] * para[1].T * y_1[j].T
        else:
            print('Errrrror')
            break
        dy[j+p,:] = delta.T            
        y_1[j+1] = y_1[j] + delta.T
    b = np.mat([[b1],[b2]])
    spread = b.T*lyc[p:].T
    spread_m = np.array(b.T*y_1.T).flatten()
    return spread_m,spread


def get_Estd(stock1,stock2,w1,w2,_model,dy=True,D=16):
    model = ""
    if _model == 'model1':
        model = 'H2'
    elif _model == 'model2':
        model = 'H1*'
    elif _model == 'model3':
        model = 'H1'
    stock1 = stock1[:150]
    stock2 = stock2[:150]
    b1 = w1
    b2 = w2
    b = np.mat([[b1],[b2]])
    y = np.vstack( [stock1, stock2] ).T
    logy = np.log(y)#np.log(y)
    p = order_select(logy,5)
    u,A,_ = para_vecm(logy,model,p)
    constant = np.mat(A[:,0])
    A = A[:,1:]
    l = A.shape[1]
    extend = np.hstack([np.identity(l-2),np.zeros([l-2,2])])
    newA = np.vstack([A,extend])
    if not dy:
        lagy = logy[p-1:-1,:]
        for i in range(1,p):
            lagy = np.hstack([lagy,logy[p-1-i:-i-1,:]])
        MatrixA = np.mat(A)
        MatrixLagy = np.mat(lagy)
        Estimate_logy = MatrixA * MatrixLagy.T + constant
        e = logy[p:,:].T-Estimate_logy
        var = e*e.T/e.shape[1]
    else:
        var = u*u.T/u.shape[1]
    NowCoef = np.mat(np.eye(len(newA)))
    Evar = var.copy()
    for i in range(149):
        NowCoef = newA * NowCoef
        Evar = Evar + NowCoef[:2,:2]*var*NowCoef[:2,:2].T
    Evar = b.T * Evar * b
    # print(Evar)
    
    return np.sqrt(Evar)


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
    # alignment.plot(xlab = str(table.stock1[pair]) , ylab = str(table.stock2[pair]) ,type="threeway")
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
    new_dynamic_stock2_series = -table.w2[pair] * np.log(dynamic_stock2_series)+table.mu[pair]
    
    """
    recaculate the weight of cointegration 
    write model selection function
    """
    stock1_series = np.log(stock1_series)
    stock2_series = np.log(stock2_series)
    dynamic_stock2_series = np.log(dynamic_stock2_series)
    # print(pd.DataFrame([stock2_series,dynamic_stock2_series]))
    # print("="*100)
    tmp = cointegration_weight(stock1_series, dynamic_stock2_series)[:2]
    # print(tmp)
    w1, w2 = tmp#cointegration_weight(stock1_series, dynamic_stock2_series)
    total = abs(w1) + abs(w2)
    if total !=0 :
        w1 = w1 / total
        w2 = w2 / total
    spread_m,spread = spread_mean(stock1_series, dynamic_stock2_series,w1,w2,table.model_type[pair])
    std = get_Estd(stock1_series, dynamic_stock2_series, w1, w2, table.model_type[pair])
    

    # table.w1[pair] = w1
    # table.w2[pair] = w2
    # print(table.w1[pair], table.w2[pair],  w1, w2)
    # alignment2 = dtw(stock1_series, dynamic_stock2_series, keep_internals=True, window_type="sakoechiba", window_args={'window_size': 10})

    return w1,w2, np.mean(spread_m[-1:]), std[0][0]




if __name__ == '__main__':


    min_path_dir = "../pair_data/{}/"
    min_path = "../pair_data/{}/averageprice/"
    tick_path = "../pair_data/{}/minprice/"
    half_path = "../pair_data/{}_halfmin/"
    table_path = "../pair_data/newstdcompare{}/"
    dtw_table_path = "../pair_data/newstdcomparedtw{}/"

    formate_time = 150       # 建模時間長度
    trade_time = 100         # 回測時間長度
    j = 0
    flag = 1


    start = False
    print(sys.argv)
    for year in range(2016,2019):
        # query = "SELECT distinct(left(DateTime,10)) as td_date FROM Fintech.Stock_1Min_Price_Tick where DateTime >= '"+ str(year) +"-01-01 09:00' and DateTime <= '"+ str(year) +"-12-31 13:30';"
        # fin_cursor.execute(query)
        # result = fin_cursor.fetchall()
        # fin_db.commit()
        td_date = [f.split('_')[0] for f in os.listdir(min_path.format(year))]  #[i["td_date"] for i in result]
        td_date.sort()
        print(td_date)
        os.makedirs(dtw_table_path.format(year), exist_ok=True)
        
        
        for choose_date in td_date:
            print(year, choose_date)

            try:
                table = pd.read_csv(table_path.format(year) +"{}_table.csv".format(choose_date))
                mindata = pd.read_csv(min_path.format(year) +"{}_averagePrice_min.csv".format(choose_date))
                mindata = mindata.iloc[16:166]

                all_combinations = combinations(mindata.columns,2)

                num = np.arange(0,len(table),1)
                dtw_distance = []
                # print(table)
                for pair in num:
                    w1,w2,e_mu, std = Dynamic_Time_Warping_with_cointegration(mindata ,pair, table)
                    table.iloc[pair, table.columns.get_loc('w1')] = w1
                    table.iloc[pair, table.columns.get_loc('w2')] = w2
                    table.iloc[pair, table.columns.get_loc('Emu')] = e_mu
                    # print(table.iloc[pair, table.columns.get_loc('Estd')])
                    # table.iloc[pair, table.columns.get_loc('Estd')] = std
                # table['dtw_distance'] = dtw_distance

                # stock1_name = table.stock1.astype('str',copy=False)
                # stock2_name = table.stock2.astype('str',copy=False)
                # test_stock1 = np.array(df[stock1_name].T)
                # test_stock2 = np.array(df[stock2_name].T)
                
                # mean = np.zeros(len(table))
                # std = np.zeros(len(table))
                # for i in range(len(table)):
                #     spread_m,spread = spread_mean(test_stock1,test_stock2,i,table)

                #     mean[i] = np.mean(spread_m[-1:])
                #     #std[i] = np.sqrt(np.mean(np.square(spread_m-spread)))
                #     std[i] = get_Estd(test_stock1,test_stock2,i,table)
                # table['e_mu'] = mean
                # table['e_stdev'] = std   
                table.to_csv(dtw_table_path.format(year)+"{}_table.csv".format(choose_date), index=False)

                # print(table)
                print("save table data ......... done.")
                print("-----------------------------------------------")   
            except Exception as e :
                print("Failed to save table data", e)
                print("-----------------------------------------------")
                pass

