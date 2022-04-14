import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import datetime

start = '2010-12-15'
end = datetime.today().strftime('%Y-%m-%d')

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

data = yf.download(user_input.upper(), start, end)
data.head()
#Describing Data
st.subheader( 'Data from 2010 - 2019')
st.write(data.describe())

#Visualizations
st.subheader ('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(data.Close)
st.pyplot(fig)

#MA
st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100= data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize =(12,6)) 
plt.plot(ma100, 'g')
plt.plot(ma200, 'r')
plt.plot(data.Close)
st.pyplot(fig)


# Splitting Data into Training and Testing
data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])

data_testing = pd.DataFrame(data['Close'][int(len(data) *0.70): int(len(data))])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)



# #Splitting Data into x_train and y_train
# x_train = []
# y_train = []

# for i in range(100, data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100: i])
#     y_train.append (data_training_array[i, 0])

# x_train, y_train = np.array(x_train), np.array(y_train)


#Load my model
model =load_model('keras_model.h5')

past_100_days = data_training.tail(100)
final_data = past_100_days.append(data_testing,ignore_index=True)
input_data = scaler.fit_transform(final_data)
           
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append (input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted =y_predicted *scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')                 
plt.ylabel('Price')
plt.legend ()
st.pyplot(fig2)