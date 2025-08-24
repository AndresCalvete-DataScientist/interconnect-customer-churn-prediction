import pandas as pd
import joblib
import numpy as np
from functions.model_evaluation import evaluate_model_test


def predict(training=True):
    
    
    # 1. Loading data ---------------------------------------- 


    model_data = joblib.load(f"files/modeling_output/model_fit/c01_model.joblib")

    model = model_data['model']
    threshold = model_data['threshold']

    if type(model).__name__ == 'LogisticRegression':
        features_test = pd.read_csv("files/datasets/testing/a03_ordinal_features_test.csv")        
    else:
        features_test = pd.read_csv("files/datasets/testing/a03_ohe_features_test.csv")

    target_test = pd.read_csv("files/datasets/testing/a03_target_test.csv")

    info_test = pd.read_csv("files/datasets/testing/a03_info_test.csv")


    # 2. Apply model ---------------------------------------- 


    pred_proba = model.predict_proba(features_test)[:, 1]
    pred_target = [1 if x >= threshold else 0 for x in pred_proba]
    
    
    # 3. Evaluate model ---------------------------------------- 
    
    
    if training:
        test_results = evaluate_model_test(model, pred_proba, pred_target, target_test, threshold)  
    

    # 4. Join results ---------------------------------------- 
    
    
    predict_output = pd.concat([info_test, pd.Series(pred_target)], axis=1)
    
    
    # 5. Save data ---------------------------------------- 


    predict_output.to_csv("files/datasets/output/d01_predict_output.csv", index=False)
    
    if training:
        test_results.to_csv(f"files/modeling_output/insights/{type(model).__name__}_test.csv") 
        
    print('⚙️  Predicción completada.')


if __name__ == "__main__":
    predict()