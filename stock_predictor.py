
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data = yf.download('AAPL', start='2015-01-01', end='2024-12-31')

data = data[['Close']]


data['Next_Close'] = data['Close'].shift(-1)

data.dropna(inplace=True)

X = data[['Close']]
y = data['Next_Close']

split_index = int(len(data) * 0.8)
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

model = LinearRegression()
model.fit(X_train, y_train)

predicted = model.predict(X_test)

mse = mean_squared_error(y_test, predicted)
print("Mean Squared Error:", round(mse, 4))

plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test, label='Actual Price')
plt.plot(y_test.index, predicted, label='Predicted Price', color='orange')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
