# Librerias ----------------------------------------

import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
# Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código
sys.path.append(os.getcwd())


# 1. Loading data ----------------------------------------

data = pd.read_csv("files/datasets/intermediate/a02_data.csv")
data_ordinal = pd.read_csv("files/datasets/intermediate/a02_data_ordinal.csv")
data_ohe = pd.read_csv("files/datasets/intermediate/a02_data_ohe.csv")


# 2. Splitting data into sets ----------------------------------------

# Guardamos la columna informativa
info = data['customerID']

# Guardamos la columna objetivo
target = data['IsChurn']

# Seleccionamos las características
ordinal_features = data_ordinal.drop(labels=['customerID', 'IsChurn'], axis=1)
ohe_features = data_ohe.drop(labels=['customerID', 'IsChurn'], axis=1)

# Segmentación de los conjuntos en entrenamiento y pruebas
info_train, info_test, ordinal_features_train, ordinal_features_test, ohe_features_train, ohe_features_test, target_train, target_test = train_test_split(
    info, ordinal_features, ohe_features, target, test_size=0.2, random_state=12345
)


# 3. Save data sets----------------------------------------

info_train.to_csv(
    "files/datasets/intermediate/a03_info_train.csv", index=False)
info_test.to_csv(
    "files/datasets/intermediate/a03_info_test.csv", index=False)
ordinal_features_train.to_csv(
    "files/datasets/intermediate/a03_ordinal_features_train.csv", index=False)
ordinal_features_test.to_csv(
    "files/datasets/intermediate/a03_ordinal_features_test.csv", index=False)
ohe_features_train.to_csv(
    "files/datasets/intermediate/a03_ohe_features_train.csv", index=False)
ohe_features_test.to_csv(
    "files/datasets/intermediate/a03_ohe_features_test.csv", index=False)
target_train.to_csv(
    "files/datasets/intermediate/a03_target_train.csv", index=False)
target_test.to_csv(
    "files/datasets/intermediate/a03_target_test.csv", index=False)
