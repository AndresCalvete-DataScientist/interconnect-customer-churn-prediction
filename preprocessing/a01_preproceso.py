# Librerias ----------------------------------------

import pandas as pd
import os
import sys
from functions import categoric_to_numeric_bool
# Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código
sys.path.append(os.getcwd())

# Loading data ----------------------------------------

# Leemos los datos de información de contratos
df_contract = pd.read_csv('files/datasets/input/contract.csv')
# Leemos los datos personales de clientes
df_personal = pd.read_csv('files/datasets/input/personal.csv')
# Leemos los datos de los servicios de internet
df_internet = pd.read_csv('files/datasets/input/internet.csv')
# Leemos los datos de los servicios telefónicos
df_phone = pd.read_csv('files/datasets/input/phone.csv')

# Cleaning columns ----------------------------------------


def clean_columns_contract(dataset):
    # Cambio de tipo de datos de la columna 'BeginDate' a datetime
    dataset['BeginDate'] = pd.to_datetime(dataset['BeginDate']).dt.date

    # Creación de la columna 'IsChurned'
    dataset['IsChurn'] = dataset['EndDate'].map(
        lambda x: 1 if x != 'No' else 0)

    # Cambio de tipo de datos de la columna 'EndDate' a datetime
    dataset['EndDate'] = pd.to_datetime(
        dataset['EndDate'], format='%Y-%m-%d %H:%M:%S', errors='coerce').dt.date

    # Conversión de columna categórica PaperlessBilling a numérica booleana
    dataset['PaperlessBilling'] = categoric_to_numeric_bool(
        dataset['PaperlessBilling'])

    # Reemplazo de valores no numéricos en 'TotalCharges' por 0.0
    dataset.loc[dataset['TotalCharges'].str.contains(
        '[^\d.]'), 'TotalCharges'] = 0.0

    # Conversión de la columna 'TotalCharges' a tipo numérico
    dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'])

    return dataset


def clean_columns_personal(dataset):
    # Conversión de columna categórica Partner a numérica booleana
    dataset['Partner'] = categoric_to_numeric_bool(dataset['Partner'])

    # Conversión de columna categórica Dependents a numérica booleana
    dataset['Dependents'] = categoric_to_numeric_bool(dataset['Dependents'])

    return dataset


def clean_columns_internet(dataset):
    # Conversión de columnas categóricas a numéricas booleana
    for col in dataset.columns.drop(['customerID', 'InternetService']):
        dataset[col] = categoric_to_numeric_bool(dataset[col])

    return dataset


def clean_columns_phone(dataset):
    # Conversión de columna categórica MultipleLines a numérica booleana
    dataset['MultipleLines'] = categoric_to_numeric_bool(
        dataset['MultipleLines'])

    return dataset


df_contract = clean_columns_contract(df_contract)
df_personal = clean_columns_personal(df_personal)
df_internet = clean_columns_internet(df_internet)
df_phone = clean_columns_phone(df_phone)

# Eliminating duplicates per period ----------------------------------------

# ...

# Checking NAs ----------------------------------------

# ...

# Guardar datos ----------------------------------------

df_contract.to_csv(
    "files/datasets/intermediate/a01_contract_cleaned.csv", index=False)
df_personal.to_csv(
    "files/datasets/intermediate/a01_personal_cleaned.csv", index=False)
df_internet.to_csv(
    "files/datasets/intermediate/a01_internet_cleaned.csv", index=False)
df_phone.to_csv(
    "files/datasets/intermediate/a01_phone_cleaned.csv", index=False)
