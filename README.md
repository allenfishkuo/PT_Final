# PT_Final

### DL_trading.py為main主程式
```
if __name__=='__main__':
    choose = 1 #決定訓練還是測試
    if choose == 0 :
        train_data, train_label, test_data, test_label = new_dataloader.read_data() #會從new_dataloader.py or new_dataloader_return.py or new_dataload_nosp.py的read_data讀取pairs的spread的mean及std資訊並處理成numpy格式回傳至DL_trading做model訓練
        loader_train,loader_test = DataTransfer(train_data, train_label, test_data, test_label)
        model_train(loader_train,loader_test)
    else :
        #find_trading_threshold.find_trading_cost_threshold()
        test.test_reward()
```


### test.py # Backtest testing
```
time = 2018 #決定哪年要做交易
cost_gate_Train = False #決定是否開啟unprofitailty detection
loading_data = False #決定是否存取資料並畫圖
```
### Actions Threshold
```
#actions = [[0.6133408446003465, 20.043565348022334], [0.6557762836185819, 9.672269763651162], [0.7123126470953718, 5.0376563854822525], [1.1433818417759427, 7.290679890624436], [1.4616844455470226, 11.091664599354965], [1.4646842029194989, 18.720335308570633], [1.5278527918781692, 16.150050761421287], [1.8079859466287909, 5.99999999999997], [1.8916951729380467, 12.846636259977194], [1.9903685727286529, 4.999999999999846], [2.030242211571373, 14.706692913385783], [2.1048266865449103, 22.69250838613482], [2.2011242403781104, 9.193923024983164], [2.9250000000000016, 20.10344827586205], [2.927334267040151, 11.825552443199511], [3.254488356362935, 7.52294907720547], [3.508635996771593, 16.51957223567391], [4.2079168858495475, 14.57469752761706], [4.323914893617021, 9.632680851063803], [4.471572794899045, 12.523910733262484], [5.022148337595906, 22.808695652173878], [6.308691275167791, 18.983221476510085], [6.350792751981882, 11.6640241600604], [6.979941239316245, 15.693376068376079], [9.98977440750324, 24.99573317561555]]
    #單純25action 0120
    #actions = [[0.61324904022712, 20.043398430980105], [0.776312126211705, 5.1989069911319366], [0.9245009219914979, 10.470937224404707], [0.9815812088226723, 7.203322830134629], [1.5091524632267055, 22.27597478402984], [1.527852791878169, 16.150050761421287], [1.5375799210991734, 18.736770507414004], [1.5718010456796814, 14.725371491469451], [1.7138786246633644, 8.829335504478017], [1.9849323562570391, 12.502254791431788], [1.9903685727286529, 4.999999999999848], [2.1222512447963604, 6.231409680842454], [2.7909482160211057, 10.338053588782445], [2.9502008928571457, 23.086383928571415], [3.0699452126271876, 8.443777719801744], [3.0746052069995753, 20.23730260349976], [3.209994443672733, 14.640088901236258], [3.5096289574511017, 16.519257914902177], [3.7828111209179194, 12.131067961165044], [3.920916046319274, 7.276054590570744], [5.781510232886389, 10.55422253587393], [5.879370395177493, 22.77494976557263], [6.24432955303536, 14.525683789192806], [6.535820895522393, 18.253731343283597], [10.00000000000039, 24.999999999999815]]
    #去除小於0.1% 在做kmeans
#actions = [[0.5228608966989476, 10.000000000000071], [0.546959896507091, 6.999999999999915], [0.6132584926132854, 20.043485518737484], [0.6351619919003917, 4.99999999999954], [0.8639708561020019, 9.000000000000012], [1.179748881153653, 6.000000000000028], [1.4149067049415627, 8.000000000000057], [1.4343396226415146, 22.348911465892662], [1.5131965006729389, 11.068640646029628], [1.5651308016877583, 14.734852320675081], [1.598206025047933, 7.000000000000076], [1.614435860582107, 18.70768954365797], [1.6283000902074771, 16.20507166482915], [1.7969422505615502, 5.000000000000077], [1.961350422832984, 13.0], [1.9823461730865428, 9.000000000000075], [2.079203959858973, 5.999999999999988], [2.2296322489391804, 9.999999999999995], [2.395085714285717, 12.000000000000005], [2.630607159039422, 23.2659719075669], [2.785531914893619, 13.999999999999998], [2.8639918116683747, 8.20081883316276], [2.867892644135189, 20.321073558648116], [2.9203545232273824, 15.316975200838316], [9.999999999999979, 24.99999999999971]]
    #去除小於0.5% 在做kmeans
actions = [[0.5, 10.0], [1.3000000000000007, 23.0], [1.3200000000000008, 6.0], [1.3500000000000008, 7.0], [1.4000000000000008, 20.0], [1.4500000000000008, 5.0], [1.4800000000000008, 11.0], [1.5000000000000009, 5.0],[1.52000000000000009, 8.0], [1.550000000000009, 7.0], [1.6500000000000008, 16.0], [1.7500000000000009, 15.0], [1.8000000000000012, 5.0], [1.8500000000000012, 9.0], [1.9500000000000013, 5.0], [2.0000000000000013, 6.0], [2.1000000000000014, 9.0], [2.200000000000001, 6.0], [2.2500000000000018, 5.0], [2.2500000000000018, 12.0], [2.4000000000000017, 10.0], [2.7500000000000018, 15.0], [2.9000000000000017, 20.0], [3.3500000000000023, 16.0], [10.0, 25.0]]
    #HighFreq 25actions 
#actions = [[0.5,2.5],[1.0,3.0],[1.5,3.5],[2.0,4.0],[2.5,4.5],[3.0,5.0]]
    #heuristic 6actions
#actions = [[0.5, 10.0], [1.3000000000000007, 23.0], [1.4500000000000008, 5.0], [2.1000000000000014, 9.0], [2.2500000000000018, 12.0], [9.881655957107576, 24.92890762493331]]
    #relabelled 6actions
```
### 模型輸入
```
Net = torch.load("????.pkl") # 輸入testing的model 
Net.eval()
whole_year = new_dataloader.test_data() #選擇輸入testing set的dataloader 可改成 return, no_sp
```

### new_dataloader
```
normalize_spread = False #設定只讀spread 還是 spread + stock price
Use_avg = False #設定使用half_min or avg_min當作Input feature來做訓練
test_period = {2016 : [path_to_2016compare , path_to_2016avg, path_to_2016half],2017 :[path_to_2017compare , path_to_2017avg, path_to_2017half],2018 : [path_to_2018compare , path_to_2018avg, path_to_2018half]}

time = 2017 #設定Loading test的年份

```
---------------------------------------------------------------------------------------------------------------------------------------------
## Tables

| Thesis Results :   | KMeans(0) | KMeans(1) | KMeans(2) | HighFreq |
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|
| 2017&2018     |  2015-2016_amsgrad_0121(15)  | 2015-2016_amsgrad_0121(M1)   | 2015-2016_amsgrad_0121(M2)    | 2015-2016_anasgrad_(M3)1      |


| Thesis Results :   | One-stage Model |Two-stage Model |
| ------------- |:-------------:|:-------------:|
| 2016      | 2013-2014_amsgrad_0120 / 2013-2014_amsgrad_0120(M3)  |  2013-2014training_stucturebreak(ST) / 2013-2014training_stucturebreak(ST)(M3)  |
| 2017      | 2014-2015_amsgrad_0120 / 2014-2015_amsgrad_0120(M3)    | 2014-2015training_stucturebreak(ST) / 2014-2015training_stucturebreak(ST)(M3)   |
| 2018      | 2015-2016_amsgrad_0121(15) / 2015-2016_anasgrad_(M3)1   | 2015-2016training_stucturebreak.pkl     |



### 尋找optimal threshold 

```
path_to_average = "./????/averageprice/"  要label的年
ext_of_average = "_averagePrice_min.csv" 
path_to_minprice = "./????/minprice/"
ext_of_minprice = "_min_stock.csv"
path_to_compare = "./newstdcompare????/"
ext_of_compare = "_table.csv"
```
find_ground_truth.py為尋找最佳化配對門檻的程式 經由check_open_loss.py選出的門檻中(個數自定義)幫每組Pair選出最佳化的門檻

### Two-stage Model :

* 首先find_trading_threshold.py會幫Training set2的Pair做Label 會賺錢設為1 不會賺錢設為0

```
path_to_threshold = "./model/2015-2016_amsgrad_0120(M3)/" #決定Training set2 的Label要放哪個File
def find_threshold_data(): # function in new_dataloader.py
    path_to_minprice = "./2017/minprice/" #決定Training set2的year
```
```
def find_trading_cost_threshold():
 Net = torch.load('./Deep_learning_model/2015-2016_anasgrad_(M3)1.pkl') #輸入訓練好的Training set1 Model來幫Training set2做Label
    Net.eval()
    val_year = new_dataloader.find_threshold_data()
    val_year = torch.FloatTensor(val_year).cuda()
```
```
if profit > 0 :
  action_ = 1
else :
  action_ = 0
```
* 接著透過DL_trading_costgate.py訓練unprofitabilty detection model
```
 torch.save(msresnet,"2015-2016training_stucturebreak(ST)(M3).pkl") #儲存訓練好的unprofitability detection model
```