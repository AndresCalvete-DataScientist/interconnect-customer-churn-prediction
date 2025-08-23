# Librerias ----------------------------------------

import params as params
import os
import sys
import argparse
# Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código
sys.path.append(os.getcwd())


# 1. Definir extension de ejecutables ----------------------------------------

if params.sistema_operativo == 'Windows':
    extension_binarios = ".exe"
else:
    extension_binarios = ""


# 2. Info ----------------------------------------

print(
    f"---------------------------------- \nComenzando proceso para entrenamiento:\n----------------------------------")


# 3. Preproceso ----------------------------------------

os.system(f"python{extension_binarios} preprocessing/a01_preprocessing.py")

os.system(f"python{extension_binarios} preprocessing/a02_encoding.py")

os.system(f"python{extension_binarios} preprocessing/a03_split_train_test.py")

# 4. Modeling creation ----------------------------------------

os.system(f"python{extension_binarios} models/c01_model_creation.py")
