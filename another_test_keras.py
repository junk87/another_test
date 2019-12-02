import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset_train = pd.read_csv('NSE-TATAGLOBAL.csv') #讀取訓練資料
training_set = dataset_train.iloc[:, 1:2].values #選擇開盤跟最高價這兩列


from sklearn.preprocessing import MinMaxScaler 
sc = MinMaxScaler(feature_range = (0, 1)) #轉換由向向量行組成的資料廖集 將特徵調整到0到1
training_set_scaled = sc.fit_transform(training_set) #資料轉換

X_train = []
y_train = []
for i in range(60, 2035):
    X_train.append(training_set_scaled[i-60:i, 0]) 
    y_train.append(training_set_scaled[i, 0]) 
X_train, y_train = np.array(X_train), np.array(y_train) #將資料儲存
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) #給陣列新形狀


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential() #keras模型初始化
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1)) #堆疊模型
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') #啟動模型
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32) #訓練模型

dataset_test = pd.read_csv('tatatest.csv') #讀取測試資料
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 76):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test) #用x的資料做預測
predicted_stock_price = sc.inverse_transform(predicted_stock_price) #將標準化後的資料還原回去


plt.plot(real_stock_price, color = 'black', label = 'TATA Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TATA Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
print('success')
plt.savefig('plot.png') #輸出程圖片
#plt.show()
