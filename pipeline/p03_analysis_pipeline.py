from preprocessing import a01_preprocessing, a02_encoding, a03_split_train_test
from insights import b01_contracts_plots, b02_personal_plots, b03_internet_plots, b04_phone_plots, b05_services_plots


def main():
        
        
    # 1. Info ----------------------------------------


    print(f"---------------------------------- \nComenzando proceso de análisis de datos:\n----------------------------------")


    # 2. Preproceso ----------------------------------------


    a01_preprocessing.preprocess()

    a02_encoding.encode()

    a03_split_train_test.split()


    # 3. Dashboards and plotting ----------------------------------------


    b01_contracts_plots.plot()  

    b02_personal_plots.plot()

    b03_internet_plots.plot()

    b04_phone_plots.plot()
    
    b05_services_plots.plot()


if __name__ == "__main__":
    main()