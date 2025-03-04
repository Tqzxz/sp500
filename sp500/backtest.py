from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd


def predict(train,test,predictors,model):
    model.fit(train[predictors],train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds,index=test.index, name="Predictions")
    return pd.concat([test["Target"],preds],axis=1)

def backtest(data,model,predictors,start=2500,step=250):
    all_predictions = []
    
    for i in range(start,data.shape[0],step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train,test,predictors,model)
        all_predictions.append(predictions)
        print(f"Back testing: Start from 2500,current_{i}:{predictions}")
    return pd.concat(all_predictions)