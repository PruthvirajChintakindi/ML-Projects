import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import datetime as dt
from keras.models import load_model
import streamlit as st
#!pip install pandas-datareader yfinance
import pandas_datareader.data as web
import yfinance as yf
import pandas as pd

start = '2011-01-01'
end = '2023-12-31'

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter ticker', 'AAPL')
# Use yfinance to fetch data as pandas_datareader might have issues with Yahoo Finance
df = yf.download(user_input, start=start, end=end)

st.subheader('Data From 2011 to 2023')
st.write(df.describe())

st.subheader('Closing Price vs Time')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

model = load_model('keras_model.h5')

past100dios = data_training.tail(100)
finaldf = pd.concat([past100dios, data_testing], ignore_index=True)
inputdata = scaler.fit_transform(finaldf)

x_test = []
y_test = []

for i in range(100, inputdata.shape[0]):
    x_test.append(inputdata[i-100: i]) 
    y_test.append(inputdata[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Predicted vs Actual')
fig1 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)



