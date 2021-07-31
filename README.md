# PT_Final
NCTU_Graduate_Thesis_Code
DL_trading.py為main主程式
會從new_dataloader.py or new_dataloader_return.py or new_dataload_nosp.py的read_data讀取pairs的spread的mean及std資訊並處理成numpy格式
回傳至DL_trading做model訓練
test過程也會從new_dataloader.py or new_dataloader_return.py or new_dataload_nosp.py的test_data讀取pairs的spread的mean及std資訊並處理成numpy格式
接著進入test.py輸入model_type及backtest程式中
loading_data = True 則會經過設定好的路徑儲存results資訊並經由MDD.py畫圖
---------------------------------------------------------------------------------------------------------------------------------------------
尋找optimal threshold code :
find_ground_truth.py為尋找最佳化配對門檻的程式 經由check_open_loss.py選出的門檻中(個數自定義)幫每組Pair選出最佳化的門檻

---------------------------------------------------------------------------------------------------------------------------------------------
Two-stage Model :
首先find_trading_threshold.py會幫Training set2的Pair做Label 會賺錢設為1 不會賺錢設為0
接著透過DL_trading_costgate.py訓練unprofitabilty detection model
---------------------------------------------------------------------------------------------------------------------------------------------
