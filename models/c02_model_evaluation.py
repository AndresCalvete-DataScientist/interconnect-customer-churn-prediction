import pandas as pd
import numpy as np
import joblib
from functions.model_evaluation import evaluate_model_cv


def evaluate():
    
    
    # 1. Loading data ----------------------------------------

    
    model = joblib.load(f"files/modeling_output/model_fit/c01_model.joblib")['model']
    
    if type(model).__name__ == 'LogisticRegression':
        features = pd.read_csv("files/datasets/training/a03_ordinal_features_train.csv")        
    else:
        features = pd.read_csv("files/datasets/training/a03_ohe_features_train.csv")
    
    target = pd.read_csv("files/datasets/training/a03_target_train.csv").values.ravel()       


    # 2. Evaluate model ----------------------------------------
    
    
    model_results = evaluate_model_cv(
        model,
        features.to_numpy(),
        target,
        cv_splits=5)
 
    
    # 3. Save results ----------------------------------------
    
    
    model_results.to_csv(f"files/modeling_output/insights/{type(model).__name__}.csv")    

    print('📑 Evaluación del modelo completada.')
    

if __name__ == "__main__":
    evaluate()
