from preprocessing import a01_preprocessing, a02_encoding, a03_split_train_test
from execution import d01_ejecucion_de_modelo


def main():


    # 1. Info ----------------------------------------

    print(f"---------------------------------- \nComenzando proceso de predicción:\n----------------------------------")


    # 2. Preprocess ----------------------------------------


    a01_preprocessing.preprocess(training=False)

    a02_encoding.encode(balance_target=False)

    a03_split_train_test.split(training=False)


    # 3. Model prediction ----------------------------------------


    d01_ejecucion_de_modelo.predict(training=False)
    

if __name__ == "__main__":
    main()
