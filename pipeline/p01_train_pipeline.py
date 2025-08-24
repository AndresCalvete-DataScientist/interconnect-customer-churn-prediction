from preprocessing import a01_preprocessing, a02_encoding, a03_split_train_test
from models import c01_model_creation, c02_model_evaluation


def main():
        
        
    # 1. Info ----------------------------------------


    print(f"---------------------------------- \nComenzando proceso de entrenamiento del modelo:\n----------------------------------")


    # 2. Preproceso ----------------------------------------


    a01_preprocessing.preprocess()

    a02_encoding.encode()

    a03_split_train_test.split()


    # 3. Modeling creation ----------------------------------------


    #! Seleccionar el modelo elegido y su umbral
    c01_model_creation.create_XGBoost_fixed_model(threshold=0.4)
    
    
    # 4. Model evaluation----------------------------------------


    c02_model_evaluation.evaluate()


if __name__ == "__main__":
    main()