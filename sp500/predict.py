from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd

def predict(stock_data):
    ## 1. Model Initialization
    model = RandomForestClassifier(n_estimators=100,min_samples_split=100,random_state=1)

    ## 2. Make train and test data-set
    ## Just do not use Cross-validation method to test our model perfermance
    train = stock_data.iloc[:-100]
    test = stock_data.iloc[-100:]

    ## 3. Create Predictor as reference for training
    predictors = ["Close","Volume","Open", "High","Low"]

    ## 4. Train model
    model.fit(train[predictors],train["Target"])

    ## 5. Generate Predictions based on Test set
    preds = model.predict(test[predictors])

    ## 6. Turn preds numpy array into pd.Series
    preds = pd.Series(preds,index=test.index)
    
    ## 7. Calculate Prediction_precision_score
    precision_scores = precision_score(test["Target"],preds)

    combined = pd.concat([test["Target"],preds],axis=1)
    try:
        print("Creating csv file for Actual labels Target and Predicted labels")
        csv_path = "data/actual_predict_compare.csv"
        combined.to_csv(csv_path,index=True)
    except FileExistsError as e:
        print("The file exists locally")
        pass

    return precision_scores

    
