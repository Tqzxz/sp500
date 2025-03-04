import yfinance as yf
import matplotlib.pyplot as plt
import os
import predict as model_predictor
import pandas as pd
import backtest as bt
from sklearn.ensemble import RandomForestClassifier

sp500 = yf.Ticker("^GSPC")

sp500 = sp500.history(period="max")

sp500.plot.line(y="Close", use_index=True)


## The graph that showing the Close prices changes in terms of time
plt.savefig("Time_Close_Prices_Graph.png")



############################ Deal with stock data

## 1. Delete unused colomns 
del sp500["Stock Splits"]
del sp500["Dividends"]

## 2. Add an additional column for indicating tomorrow close price
sp500["Tomorrow"] = sp500["Close"].shift(-1)

## 3. Create the lables called Target for machine learning algorithm
## Use boolean expression 0 for Stock price will go down tomorrow , 1 is opposite
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

## 4. Remove too far data , e.g. remove all data before 2000
sp500 = sp500.loc["2000-01-01":].copy()

## 5. Store Data locally
try:
    print("Creating csv file for all stock data from 2000 until now")
    os.mkdir("data")
    csv_path = "data/clean_data.csv"
    sp500.to_csv(csv_path,index=True)
except FileExistsError as e:
    print("The file exists locally")
    pass


## 6. Train Model and output test result without improvment
## 7. Visualize Target(actual lable) vs Predicted label
precision_scores = model_predictor.predict(sp500)

### 8. Backtest
model = RandomForestClassifier(n_estimators=100,min_samples_split=100,random_state=1)
predictors = ["Close","Volume","Open", "High","Low"]
backtest_predictions = bt.backtest(sp500,model,predictors)
print(backtest_predictions["Predictions"].value_counts())
