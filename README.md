# seq2seq_char

seq2eq利用字元級別訓練

使用方式：

    x = seq2seq_all('P1_big_BN.h5', 'QA_all.txt') #參數說明第一個參數為儲存權重黨名稱，第二個參數為輸入txt檔

    x.build_basic_model() #訓練模型

    while True:
        test_text = [input('【input Answer】 \n' )] #輸入問句
        result = x.run_model(test_text) #產生問句
        print('【output question】 \n', result) #列印問句


數據及格式：

    請問今天天氣如何\t陰天\n

