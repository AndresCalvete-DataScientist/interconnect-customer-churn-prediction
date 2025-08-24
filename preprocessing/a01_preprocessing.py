import pandas as pd
import numpy as np
from functions.categoric_to_numeric_bool import categoric_to_numeric_bool


def preprocess(training=True):
    
    
    # 1. Loading data ----------------------------------------


    # Leemos los datos de información de contratos
    df_contract = pd.read_csv('files/datasets/input/contract.csv')
    # Leemos los datos personales de clientes
    df_personal = pd.read_csv('files/datasets/input/personal.csv')
    # Leemos los datos de los servicios de internet
    df_internet = pd.read_csv('files/datasets/input/internet.csv')
    # Leemos los datos de los servicios telefónicos
    df_phone = pd.read_csv('files/datasets/input/phone.csv')


    # 2. Cleaning columns ----------------------------------------


    def clean_columns_contract(dataset):
        # Cambio de tipo de datos de la columna 'BeginDate' a datetime
        dataset['BeginDate'] = pd.to_datetime(dataset['BeginDate']).dt.date

        if training:
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
            r'[^\d.]'), 'TotalCharges'] = 0.0

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


    # 3. Feature engineering and joins ----------------------------------------


    def create_begin_year_month_column(dataset):
        # Extraemos el año y mes de la fecha de subscripción
        dataset['BeginYear'] = [x.year for x in dataset['BeginDate']]
        dataset['BeginMonth'] = [x.month for x in dataset['BeginDate']]

        return dataset

    # Calcula la duración en meses antes del abandono  
    def diff_months(row):  
        if pd.notnull(row['EndDate']) and pd.notnull(row['BeginDate']):  
            end = row['EndDate']
            begin = row['BeginDate']
            return (end.year - begin.year) * 12 + (end.month - begin.month)  
        else:
            return np.nan

    # Creamos las columnas de año y mes de inscripción
    df_contract = create_begin_year_month_column(df_contract)
    
    if training:
        # Aplica la función para calcular la duración en meses
        df_contract['DurationMonths'] = df_contract.apply(diff_months, axis=1)

    # Configuramos una tabla con todos los servicios
    df_con_int = df_contract.merge(df_internet, how='left', on='customerID')
    
    # Resumir la columna InternetService en una nueva columna booleana HasInternet 
    df_con_int['HasInternet'] = [1 if pd.notnull(x) else 0 for x in df_con_int['InternetService']]
    
    # Configuramos una tabla con todos los servicios
    df_con_int_pho = df_con_int.merge(df_phone, how='left', on='customerID')

    # Resumir la columna MultipleLines en una nueva columna booleana HasPhone
    df_con_int_pho['HasPhone'] = [1 if pd.notnull(x) else 0 for x in df_con_int_pho['MultipleLines']]

    # Crear una nueva columna que cuente la cantidad de servicios activos por cliente
    all_service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'HasInternet', 'HasPhone']
    df_con_int_pho['AllServicesCount'] = df_con_int_pho[all_service_columns].sum(
        axis=1).astype(int)

    # Selección de características y objetivos
    data_columns = [
        df_contract[[
            'customerID',
            'BeginYear',
            'BeginMonth',
            'Type',
            'PaperlessBilling',
            'PaymentMethod',
            'MonthlyCharges',
            'TotalCharges',
        ]],
        df_personal[[
            'SeniorCitizen',
            'Partner',
            'Dependents'
        ]],
        df_con_int_pho[[
            'InternetService',
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'AllServicesCount'
        ]]
    ]
    
    if training:
        data_columns.append(df_contract[['IsChurn']])
    
    data = pd.concat(
        data_columns,
        axis=1
    )


    # 4. Checking NAs ----------------------------------------


    # Reemplazamos los nulos de InternetService con 'No'
    data['InternetService'] = data['InternetService'].fillna('No')

    # Reemplazamos los nulos de los servicios con 0 significando que no se posee el servicio
    data['OnlineSecurity'] = data['OnlineSecurity'].fillna(0)
    data['OnlineBackup'] = data['OnlineBackup'].fillna(0)
    data['DeviceProtection'] = data['DeviceProtection'].fillna(0)
    data['TechSupport'] = data['TechSupport'].fillna(0)


    # 5. Saving data ----------------------------------------


    df_contract.to_csv(
        "files/datasets/intermediate/a01_contract_cleaned.csv", index=False)
    df_personal.to_csv(
        "files/datasets/intermediate/a01_personal_cleaned.csv", index=False)
    df_internet.to_csv(
        "files/datasets/intermediate/a01_internet_cleaned.csv", index=False)
    df_phone.to_csv(
        "files/datasets/intermediate/a01_phone_cleaned.csv", index=False)
    data.to_csv(
        "files/datasets/intermediate/a01_data_cleaned.csv", index=False)
    
    print("✅ Preprocesamiento finalizado.")


if __name__ == "__main__":
    preprocess()
