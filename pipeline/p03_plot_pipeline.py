from preprocessing import a01_preprocessing, a02_encoding, a03_split_train_test
from insights import b01_contracts_plots

def main():
        
        
    # 1. Info ----------------------------------------


    print(f"---------------------------------- \nComenzando proceso de visualización:\n----------------------------------")


    # 2. Preproceso ----------------------------------------


    a01_preprocessing.preprocess()

    a02_encoding.encode()

    a03_split_train_test.split()


    # 3. Dashboards and plotting ----------------------------------------


    b01_contracts_plots.plot()  

    # os.system(f"python{extension_binarios} preprocessing/b02_personal_plots.py")

    # os.system(f"python{extension_binarios} preprocessing/b03_internet_plots.py")

    # os.system(f"python{extension_binarios} preprocessing/b04_phone_plots.py")


if __name__ == "__main__":
    main()