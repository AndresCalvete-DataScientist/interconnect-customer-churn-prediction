# Librerias ----------------------------------------

import params as params
import os
import sys
import argparse
# Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código
sys.path.append(os.getcwd())


# 1. Argumentos por linea de comandos ----------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    '--periodo', default=f'{params.periodo_YYYYMM}', help='periodo en formato YYYYMM')

try:
    args = parser.parse_args()
except argparse.ArgumentTypeError as e:
    print(f"Invalid argument: {e}")


# 2. Definir extension de ejecutables ----------------------------------------

if params.sistema_operativo == 'Windows':
    extension_binarios = ".exe"
else:
    extension_binarios = ""


# 3. Info ----------------------------------------

print(
    f"---------------------------------- \nComenzando proceso para periodo: {args.periodo}\n----------------------------------")


# 4. Preproceso ----------------------------------------

os.system(f"python{extension_binarios} preprocessing/a01_preprocessing.py")

os.system(f"python{extension_binarios} preprocessing/a02_encoding.py")

os.system(f"python{extension_binarios} preprocessing/a03_split_train_test.py")


# 5. Dashboards and plotting ----------------------------------------

os.system(f"python{extension_binarios} preprocessing/b01_contracts_plots.py")

os.system(f"python{extension_binarios} preprocessing/b02_personal_plots.py")

os.system(f"python{extension_binarios} preprocessing/b03_internet_plots.py")

os.system(f"python{extension_binarios} preprocessing/b04_phone_plots.py")


# 6. Modeling creation ----------------------------------------

os.system(f"python{extension_binarios} models/c01_model_creation.py")
