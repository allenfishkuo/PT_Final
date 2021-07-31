# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:58:29 2021

@author: allen
"""
import os 
import pandas as pd
import numpy as np
path_to_2017compare = "./newstdcompare2017/"
ext_of_compare = "_table.csv"
path_to_minprice = "./2017/minprice/"
ext_of_minprice = "_min_stock.csv"
path_to_average = "./2017/averageprice/"
ext_of_average = "_averagePrice_min.csv"


import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
#from Predict_Client import send_request
from scipy.stats import skew
from cost import tax , slip 
from integer import num_weight
from vecm import rank
from MTSA import fore_chow , spread_chow , order_select
#import tensorflow
#from keras.models import load_model
import pandas as pd
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.family']='sans-serif' 
plt.rcParams['axes.unicode_minus'] = False
# 標準差倍數當作停損門檻(滑價＋交易稅)-------------------------------------------------------------------------------
path_to_image = "./total_profit_plot/"
def two_pairs(pair, formate_time, table, min_data, tick_data, up_open_time, down_open_time, stop_loss_time, day1, maxi,
          tax_cost, cost_gate, capital,  dump=False):
    '''
    if str(table.stock1[pair]) == '1504' and str(table.stock2[pair]) == '2301':
        pass
    else:
        return 0,0,0,[0,0,0]
    '''
    #table = pd.DataFrame(table).T
    trade_capital = 0
    cpA,cpB = 0,0
    trading =[0,0,0]
    weight = [0,0]
    '''
    if table.model_type[pair] == 'model1':
        model_name = 'H2'
    elif table.model_type[pair] == 'model2':
        model_name = 'H1*'
    else:
        model_name = 'H1'
    '''
    use_fore_lag5 = False
    use_adf = False
    # 波動太小的配對不開倉
    if up_open_time *table.Estd[pair] < cost_gate:
        trading_profit = 0
        trade = 0
        local_profit = 0
        local_open_num = 0
        local_rt = 0
        local_std = 0
        local_skew = 0
        local_timetrend = 0
        position = 0
        return local_profit, local_open_num, trade_capital,trading,weight
    min_price = day1
    # min_price = min_price.dropna(axis = 1)
    # min_price.index  = np.arange(0,len(min_price),1)
    # num = np.arange(0,len(table),1)
    t = formate_time  # formate time
    stock1_seq = min_price[str(table.stock1[pair])].loc[0:t]
    stock2_seq = min_price[str(table.stock2[pair])].loc[0:t]
    # z = ( np.vstack( [stock1_seq , stock2_seq] ).T )
    # p = order_select(z,5)
    local_open_num = []
    local_profit = []
    local_rt = []
    local_std = []
    local_skew = []
    local_timetrend = []
    # for pair in num:
    # spread = table.w1[pair] * np.log(min_data[ table.stock1[pair] ]) + table.w2[pair] * np.log(min_data[ table.stock2[pair] ])
    spread = table.w1[pair] * np.log(tick_data[str(table.stock1[pair])]) + table.w2[pair] * np.log(
        tick_data[str(table.stock2[pair])])
    #print("spread length :",len(spread))
    up_open = table.Emu[pair] + table.Estd[pair] * up_open_time  # 上開倉門檻
    down_open = table.Emu[pair] - table.Estd[pair] * down_open_time  # 下開倉門檻
    stop_loss = table.Estd[pair] * stop_loss_time  # 停損門檻
    close = table.Emu[pair]  # 平倉(均值)
    #M = round(1 / table.zcr[pair])  # 平均持有時間
    trade = 0  # 計算開倉次數
    break_point = 0  # 計算累積斷裂點
    # break_CNN = 0
    # discount = 1
    position = 0  # 持倉狀態，1:多倉，0:無倉，-1:空倉，-2：強制平倉
    pos = [0]
    stock1_profit = []
    stock2_profit = []
    num = 0
    for i in range(0, len(spread) - 2):
        stock1_seq = min_price[str(table.stock1[pair])].loc[0:t + i]
        stock2_seq = min_price[str(table.stock2[pair])].loc[0:t + i]
        # z = ( np.vstack( [stock1_seq , stock2_seq] ).T )
        # r = rank( pd.DataFrame(z) , model_name , p )
        # if position == 0 and len(spread) - i > M:  # 之前無開倉
        if position == 0 and i != len(spread) - 3 and i < 40:  # 之前無開倉
        #if position == 0 and i != len(spread) - 3 :
            if  spread[i] < (close + stop_loss) and  spread[i] > up_open :  # 碰到上開倉門檻且小於上停損門檻
                # 資金權重轉股票張數，並整數化
                w1, w2 = num_weight(table.w1[pair], table.w2[pair], tick_data[str(table.stock1[pair])][i],
                                    tick_data[str(table.stock2[pair])][i], maxi, capital)
                weight[0],weight[1] = w1 ,w2
                # print(str(table.stock1[pair]),str(table.stock2[pair]),w1,w2,tick_data[str(table.stock1[pair])][(i)],tick_data[str(table.stock2[pair])][(i)])

                spread1 = w1 * np.log(stock1_seq) + w2 * np.log(stock2_seq)
                if use_adf and adfuller(spread1, regression='c')[1] > 0.05:  # spread平穩才開倉
                    position = 0
                    stock1_payoff = 0
                    stock2_payoff = 0
                else:
                    position = -1
                    stock1_payoff = w1 * slip(tick_data[str(table.stock1[pair])][(i)], table.w1[pair])
                    stock2_payoff = w2 * slip(tick_data[str(table.stock2[pair])][(i)], table.w2[pair])
                    stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                    cpA,cpB = stock1_payoff, stock2_payoff
                    # down_open = table.mu[pair] - table.stdev[pair] * close_time
                    trade = trade + 1
            elif spread[i] > (close - stop_loss) and spread[i] < down_open:  # 碰到下開倉門檻且大於下停損門檻
                # 資金權重轉股票張數，並整數化
                w1, w2 = num_weight(table.w1[pair], table.w2[pair], tick_data[str(table.stock1[pair])][(i)],
                                    tick_data[str(table.stock2[pair])][(i)], maxi, capital)
                weight[0],weight[1] = w1 ,w2
                # print(str(table.stock1[pair]), str(table.stock2[pair]),w1,w2,tick_data[str(table.stock1[pair])][(i)],tick_data[str(table.stock2[pair])][(i)])

                spread1 = w1 * np.log(stock1_seq) + w2 * np.log(stock2_seq)
                if use_adf and adfuller(spread1, regression='c')[1] > 0.05:  # spread平穩才開倉
                    position = 0
                    stock1_payoff = 0
                    stock2_payoff = 0
                else:
                    position = 1
                    stock1_payoff = -w1 * slip(tick_data[str(table.stock1[pair])][(i)], -table.w1[pair])
                    stock2_payoff = -w2 * slip(tick_data[str(table.stock2[pair])][(i)], -table.w2[pair])
                    stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                    cpA,cpB = stock1_payoff,stock2_payoff
                    # up_open = table.mu[pair] + table.stdev[pair] * close_time
                    trade = trade + 1
            else:
                position = 0
                stock1_payoff = 0
                stock2_payoff = 0
        elif position == -1:  # 之前有開空倉，平空倉
            spread1 = table.w1[pair] * np.log(stock1_seq) + table.w2[pair] * np.log(stock2_seq)
            #num = fore_chow(min_price[str(table.stock1[pair])].loc[0:t], min_price[str(table.stock2[pair])].loc[0:t], stock1_seq,
                            #stock2_seq, table.model_type[pair])  #做forlag5
            if use_fore_lag5:
                if num == 0:
                    break_point = 0
                else:  # num == 1
                    break_point = break_point + num

            if i == (len(spread) - 3):  # 回測結束，強制平倉
                position = -4
                stock1_payoff = -w1 * slip(tick_data[str(table.stock1[pair])][i], -table.w1[pair])
                stock2_payoff = -w2 * slip(tick_data[str(table.stock2[pair])][i], -table.w2[pair])
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[2] += 1
            elif (spread[i] - close) < 0:  # 空倉碰到下開倉門檻即平倉
                position = 666  # 平倉
                stock1_payoff = -w1 * slip(tick_data[str(table.stock1[pair])][i], -table.w1[pair])
                stock2_payoff = -w2 * slip(tick_data[str(table.stock2[pair])][i], -table.w2[pair])
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                # print(str(table.stock1[pair]),str(table.stock2[pair]),w1,w2,tick_data[str(table.stock1[pair])][(i)],tick_data[str(table.stock2[pair])][(i)])

                trading[0]+=1
                # down_open = table.mu[pair] - table.stdev[pair] * open_time
                # 每次交易報酬做累加(最後除以交易次數做平均)
            elif spread[i] > (close + stop_loss):  # 空倉碰到上停損門檻即平倉停損
                position = -2  # 碰到停損門檻，強制平倉
                stock1_payoff = -w1 * slip(tick_data[str(table.stock1[pair])][(i)], -table.w1[pair])
                stock2_payoff = -w2 * slip(tick_data[str(table.stock2[pair])][(i)], -table.w2[pair])
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[1]+=1
                # 每次交易報酬做累加(最後除以交易次數做平均)
 
            elif break_point == 5:
                position = -3  # 結構性斷裂，強制平倉
                stock1_payoff = -w1 * slip(tick_data[str(table.stock1[pair])][(i + 1)], -table.w1[pair])
                stock2_payoff = -w2 * slip(tick_data[str(table.stock2[pair])][(i + 1)], -table.w2[pair])
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[2]+=1

            else:
                position = -1
                stock1_payoff = 0
                stock2_payoff = 0
        elif position == 1:  # 之前有開多倉，平多倉
            spread1 = table.w1[pair] * np.log(stock1_seq) + table.w2[pair] * np.log(stock2_seq)
            #num = fore_chow(min_price[str(table.stock1[pair])].loc[0:t], min_price[str(table.stock2[pair])].loc[0:t], stock1_seq,
                           # stock2_seq, table.model_type[pair]) #做forlag5
            if use_fore_lag5:
                if num == 0:
                    break_point = 0
                else:  # num == 1
                    break_point = break_point + num

            if i == (len(spread) - 3):  # 回測結束，強制平倉
                position = -4
                stock1_payoff = w1 * slip(tick_data[str(table.stock1[pair])][i], table.w1[pair])
                stock2_payoff = w2 * slip(tick_data[str(table.stock2[pair])][i], table.w2[pair])
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[2] += 1
                # print(str(table.stock1[pair]), str(table.stock2[pair]), w1, w2, tick_data[str(table.stock1[pair])][(i)],
                #      tick_data[str(table.stock2[pair])][(i)])
                # 每次交易報酬做累加(最後除以交易次數做平均)
            elif (spread[i] - close) > 0:
                position = 666  # 平倉
                stock1_payoff = w1 * slip(tick_data[str(table.stock1[pair])][(i)], table.w1[pair])
                stock2_payoff = w2 * slip(tick_data[str(table.stock2[pair])][(i)], table.w2[pair])
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                # print(str(table.stock1[pair]),str(table.stock2[pair]),w1,w2,tick_data[str(table.stock1[pair])][(i)],tick_data[str(table.stock2[pair])][(i)])

                trading[0]+=1
                # up_open = table.mu[pair] + table.stdev[pair] * open_time
                # 每次交易報酬做累加(最後除以交易次數做平均)
            elif spread[i] < (close - stop_loss):
                position = -2  # 碰到停損門檻，強制平倉
                stock1_payoff = w1 * slip(tick_data[str(table.stock1[pair])][(i)], table.w1[pair])
                stock2_payoff = w2 * slip(tick_data[str(table.stock2[pair])][(i)], table.w2[pair])
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[1]+=1
                # 每次交易報酬做累加(最後除以交易次數做平均)

            elif break_point == 5:
                position = -3  # 結構性斷裂，強制平倉
                stock1_payoff = w1 * slip(tick_data[str(table.stock1[pair])][(i)], table.w1[pair])
                stock2_payoff = w2 * slip(tick_data[str(table.stock2[pair])][(i)], table.w2[pair])
                stock1_payoff, stock2_payoff = tax(stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[2] += 1

                position = 1
                stock1_payoff = 0
                stock2_payoff = 0
        else:
            # -4: 強迫平倉 -3: 結構性斷裂平倉(for lag 5) -2:停損 666:正常平倉
            if position == -2 or position == -3 or position == -4 or position == 666:
                stock1_payoff = 0
                stock2_payoff = 0
            else:
                position = 0  # 剩下時間少於預期開倉時間，則不開倉，避免損失
                stock1_payoff = 0
                stock2_payoff = 0
        pos.append(position)
        stock1_profit.append(stock1_payoff)
        stock2_profit.append(stock2_payoff)
    trading_profit = sum(stock1_profit) + sum(stock2_profit)

    
    if trading_profit != 0 and position == 666:
        position = 666
    # if (open_time + close_time) * table.stdev[pair] < cost_gate:
    # trading_profit = 0
    # trade = 0
    # local_profit.append(trading_profit)
    local_profit = trading_profit
    # local_open_num.append(trade)
    local_open_num = trade
    if trade == 0:  # 如果都沒有開倉，則報酬為0
        # local_rt.append(0)
        # local_std.append(0)
        # local_skew.append(0)
        # local_timetrend.append(0)
        local_rt = 0
        local_std = 0
        local_skew = 0
        local_timetrend = 0
        position = 0
    else:  # 計算平均報酬
        # local_rt.append(trading_profit/(capital*trade) )
        spread2 = w1 * np.log(min_price[str(table.stock1[pair])].iloc[0:t]) + w2 * np.log(
            min_price[str(table.stock2[pair])].iloc[0:t])
        # local_std.append(np.std(spread2))
        # local_skew.append(skew(spread2))
        x = np.arange(0, t)
        b1, b0 = np.polyfit(x, spread2, 1)
        # local_timetrend.append(b1)
        local_rt = trading_profit / (capital * trade)
        local_std = np.std(spread2)
        local_skew = skew(spread2)
        local_timetrend = b1
        # x = np.arange(0,121)
        # plt.plot(spread)
        # plt.axhline(y=close,color='r')
        # plt.axhline(y=up_open,color='r')
        # plt.axhline(y=down_open,color='r')
        # plt.axhline(y=close+stop_loss,color='green')
        # plt.axhline(y=close-stop_loss,color='green')
        # bp = np.array(np.where( np.array(pos) == -3 ))
        # if bp.size != 0:
        # plt.axvline(x=bp[0][0],color='green')
        # plt.show()
        # print([table.stock1[pair],table.stock2[pair],trading_profit])
    # posi = pos[len(spread)-2]
    '''
    if tax_cost == 0:
    
        local_profit = pd.DataFrame(local_profit)       ; local_profit.columns = ["profit without cost"]
        
    else:
        
        local_profit = pd.DataFrame(local_profit)       ; local_profit.columns = ["profit"]
        
    local_open_num = pd.DataFrame(local_open_num)   ; local_open_num.columns = ["open number"]
    local_rt = pd.DataFrame(local_rt)               ; local_rt.columns = ["return"]
    
    #back_test = pd.concat([local_profit,local_open_num,local_rt],axis=1)
    '''
    if cpA > 0 and cpB > 0:
        trade_capital = abs(cpA)+abs(cpB)
    elif cpA > 0 and cpB < 0 :
        trade_capital = abs(cpA)+0.9*abs(cpB)
    elif cpA < 0 and cpB > 0 :
        trade_capital = 0.9*abs(cpA)+abs(cpB)
    elif cpA < 0 and cpB < 0 :
        trade_capital = 0.9*abs(cpA)+0.9*abs(cpB)
    # -4: 強迫平倉 -3: 結構性斷裂平倉(for lag 5) -2:停損 666:正常平倉

    # table.stock1 , table.stock2 , local_profit , local_open_num , local_rt , local_std , local_skew , local_timetrend
    # #, 0
    return   local_profit, local_open_num, trade_capital, trading, weight

def main():
    choose = 1
    datelist = [f.split('_')[0] for f in os.listdir(path_to_2017compare)]
    print(datelist)

    dictionary_stock = {}
    top = [['2891','2308'],['3481','2409'],['1605','1303'],['3231','2883'],['2883','2308'],['6505','2105'],['2892','2002'],['2892','2882'],['2881','2344'],['2834','2002']]
    lis  =[]
    for stock in top :
        print(stock[0],stock[1])
        for date in sorted(datelist[:]): #決定交易要從何時開始
            table = pd.read_csv(path_to_2017compare+date+ext_of_compare)
            mindata = pd.read_csv(path_to_average+date+ext_of_average)
            try:
                tickdata = pd.read_csv(path_to_minprice+date+ext_of_minprice).drop([266, 267, 268, 269, 270])
            except:
                continue
            #halfmin = pd.read_csv(path_to_half+date+ext_of_half)
            #print(tickdata.shape)
            tickdata = tickdata.iloc[166:]
            tickdata.index = np.arange(0,len(tickdata),1)  
            num = np.arange(0,len(table),1)
            #print(date)
            for pair in num: #看table有幾列配對 依序讀入
                if (str(table.stock1[pair]) ==stock[0] and str(table.stock2[pair]) == stock[1]) or (str(table.stock2[pair]) ==stock[0] and str(table.stock1[pair]) == stock[1]):
                    local_profit, local_open_num, trade_capital, trading, weight= two_pairs( pair ,166,  table , mindata , tickdata , 1.5 ,1.5, 1000 ,mindata, 5 , 0.0015, 0.0015, 300000000 )
                    print(weight)

if __name__ == "__main__":
    main()