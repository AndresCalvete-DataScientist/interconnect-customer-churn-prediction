import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import joblib


def create_logistic_regression_model(threshold=0.5):
    
    
    # 1. Loading data ----------------------------------------  
    
    
    ordinal_features_train = pd.read_csv("files/datasets/training/a03_ordinal_features_train.csv")
    
    target_train = pd.read_csv("files/datasets/training/a03_target_train.csv").values.ravel()
    
    
    # 2. Training model ----------------------------------------  
    
    
    lr_model = LogisticRegression(random_state=12345)
    
    lr_model.fit(ordinal_features_train, target_train)
    
    
    # 3. Save model ----------------------------------------
    
    
    joblib.dump(
    {'model': lr_model, 'threshold':threshold},
    f"files/modeling_output/model_fit/c01_model.joblib")
    
    print("📚 Entrenamiento de modelo Logistic Regression completado.")


def create_random_forest_classifier(threshold=0.5):
    
    
    # 1. Loading data ----------------------------------------  
    
    
    ohe_features_train = pd.read_csv("files/datasets/training/a03_ohe_features_train.csv")
    
    target_train = pd.read_csv("files/datasets/training/a03_target_train.csv").values.ravel()
    
    
    # 2. Tuning model ----------------------------------------  
    
    
    # Realizando estudio de hiperparámetros basado en la exactitud.
    best_acc = 0
    best_est = 0
    best_depth = 0

    for est in range(5, 105, 5):
        for depth in range(1, 11, 1):
            rfc_model = RandomForestClassifier(n_estimators=est, max_depth=depth, random_state=12345)
            
            rfc_model.fit(ohe_features_train, target_train)
            
            acc = cross_val_score(
                rfc_model, 
                ohe_features_train, 
                target_train, 
                cv=5,
                scoring='accuracy'
            )  
            
            acc = acc.mean()
                
            if acc > best_acc:
                best_acc = acc
                best_est = est
                best_depth = depth

    print('Resultado de estudio de hiperparámetros:')
    print('Mejor exactitud:', best_acc)
    print(f'Dado con {best_est} estimadores de profundidad máxima {best_depth}')
    
    
    # 3. Training model ---------------------------------------- 
    
    
    # Entrenamos en base a los resultados de los hiperparámetros
    rfc_model = RandomForestClassifier(n_estimators=best_est, max_depth=best_depth, random_state=12345)

    rfc_model.fit(ohe_features_train, target_train)
    
    
    # 4. Save model ----------------------------------------
    
    
    joblib.dump(
    {'model': rfc_model, 'threshold':threshold},
    f"files/modeling_output/model_fit/c01_model.joblib")
    
    print("📚 Entrenamiento de modelo Random Forest completado.")


def create_XGBoost_fixed_model(threshold=0.5):
    
    
    # 1. Loading data ----------------------------------------  
    
    
    ohe_features_train = pd.read_csv("files/datasets/training/a03_ohe_features_train.csv")
    
    target_train = pd.read_csv("files/datasets/training/a03_target_train.csv").values.ravel()
    
    
    # 2. Training model ---------------------------------------- 
    
    
    xg_model = XGBClassifier(max_depth=4, min_child_weight=5, gamma=1, reg_lambda=1, random_state=12345)

    xg_model.fit(ohe_features_train, target_train)

    # 3. Save model ----------------------------------------
    
    
    joblib.dump(
    {'model': xg_model, 'threshold':threshold},
    f"files/modeling_output/model_fit/c01_model.joblib")

    print("📚 Entrenamiento de modelo XGBoost completado.")