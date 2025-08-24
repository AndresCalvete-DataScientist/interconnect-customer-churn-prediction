import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder


def encode(balance_target=True):


    # 1. Loading data ----------------------------------------


    data = pd.read_csv("files/datasets/intermediate/a01_data_cleaned.csv")


    # 2. Data balancing ----------------------------------------
    
    
    if balance_target:
        # Identificamos las observaciones de clase 1 y clase 0
        one_class = data[data['IsChurn'] == 1]
        zero_class = data[data['IsChurn'] == 0]

        # Contamos la diferencia de cantidades entre una clase y otra
        difference = zero_class['IsChurn'].count() - one_class['IsChurn'].count()

        # Aplicamos sobremuestreo con reemplazo de la clase 1 aumentando la mitad de la diferencia
        new_ones = one_class.sample(difference//2, replace=True, random_state=12345)

        # Aplicamos submuestreo de la clase 0 reduciendo la mitad de la diferencia
        new_zeros = zero_class.sample(
            len(zero_class)-(difference//2), replace=False, random_state=12345)

        data = pd.concat([one_class, new_ones, new_zeros])
        data = data.sample(frac=1, replace=False,
                        random_state=12345, ignore_index=True)


    # 3. Scaling data ----------------------------------------


    # Estandarización de características
    columns_to_standarize = ['BeginYear', 'BeginMonth',
                            'MonthlyCharges', 'TotalCharges', 'AllServicesCount']

    # Instanciar y entrenar el estandarizador
    scaler = StandardScaler()
    scaler.fit(data[columns_to_standarize])

    # Estandarizamos las columnas
    data[columns_to_standarize] = scaler.transform(data[columns_to_standarize])


    # 4. Ordinal Encoding ----------------------------------------


    # Columnas a transformar con codificación
    columns_to_code = ['Type', 'PaymentMethod', 'InternetService']

    # Creación de dataframe con caractirizticas Ordinales
    data_ordinal = data.copy()

    encoder = OrdinalEncoder()
    data_ordinal[columns_to_code] = encoder.fit_transform(
        data_ordinal[columns_to_code])


    # 5. One-Hot-Encoding ----------------------------------------


    # Creación de dataframe con caractirizticas OHE
    data_ohe = data.copy()

    # Codificación OHE
    OHE = pd.get_dummies(data_ohe[columns_to_code], drop_first=True)

    # Inserción de las columnas codificadas
    data_ohe.drop(labels=columns_to_code, inplace=True, axis=1)
    data_ohe = pd.concat([data_ohe, OHE], axis=1)


    # 6. Saving data ----------------------------------------


    data.to_csv(
        "files/datasets/intermediate/a02_data.csv", index=False)
    data_ordinal.to_csv(
        "files/datasets/intermediate/a02_data_ordinal.csv", index=False)
    data_ohe.to_csv(
        "files/datasets/intermediate/a02_data_ohe.csv", index=False)

    print("✅ Codificación finalizada.")


if __name__ == "__main__":
    encode()