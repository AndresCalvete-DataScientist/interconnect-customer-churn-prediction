# 📉 Predicción de abandono en clientes de Interconnect

## 📌 Introducción

Este proyecto aborda el problema de predicción de cancelación de clientes (churn) en la empresa de telecomunicaciones Interconnect.
El objetivo es analizar los datos de los clientes y entrenar un modelo de machine learning capaz de identificar patrones asociados a la pérdida de clientes, de modo que la empresa pueda implementar estrategias de retención más efectivas.

---

## 🎯 Objetivos

- Analizar el comportamiento de los clientes y detectar factores que influyen en el churn.
- Preprocesar y transformar los datos en un formato apto para modelado.
- Entrenar, validar y comparar diferentes modelos predictivos.
- Seleccionar el modelo más robusto con base en métricas de evaluación.
- Proveer conclusiones prácticas que apoyen la toma de decisiones.
- Crear un flujo de datos (pipeline) para la ingesta de datos y retorno de predicciones.

---

## 📂 Estructura del repositorio

```bash
interconnect-customer-churn-prediction/
│── execution/           # Scripts para la predicción con el modelo entrenado (producción y entrenamiento).
│── files/               # Carpeta reservada para datasets, modelos entrenados y resultados de análisis de modelos.
│── functions/           # Funciones auxiliares utilizadas en el proyecto.
│── insights/            # Scripts para generar dashboards y análisis de los datos de input.
│── models/              # Creación y evaluación de modelos.
│── notebooks/           # Jupyter Notebooks con el análisis exploratorio inicial y modelado exploratorio.
│── pipeline/            # Scripts gestores de los 3 principales flujos de datos: análisis, entrenamiento de modelos y predicción.
│── preprocessing/       # Scripts de limpieza y transformación de datos.
│── reports/             # Gráficas y reportes generados para la toma de decisiones.
│── README.md            # Documentación principal del proyecto
│── requirements.txt     # Librerías necesarias
│── run.py               # Script principal para la ejecución del proyecto a modo de pipelines.
```

---

## ⚙️ Instalación y requisitos

1. Clonar este repositorio:

```bash
git clone https://github.com/AndresCalvete-DataScientist/interconnect-customer-churn-prediction.git
cd interconnect-customer-churn-prediction
```

2. Crear entorno virtual e instalar dependencias:
```bash
pip install -r requirements.txt
```

## ▶️ Guía de uso

### Modo exploración en Jupyter Notebook

Este modo permite revisar el proyecto desde un Jupyter Notebook. Es una forma más sencilla y amigable con el usuario. Para esto, una vez instalados los requisitos se puede acceder a la carpeta `notebooks` e ir ejecutando las celdas en orden del archivo `eda.ipynb`.

En tal documento se encontrará una guía paso a paso y explicaciones de las decisiones tomadas para el desarrollo de la solución. Es el método recomendado para reconocer la arquitectura completa del proyecto.


### Modo producción

1. Colocar los datasets en la carpeta `files/datasets/input`.

2. Abrir la consola en la carpeta raiz.

3. Ejecutar el pipeline deseado:
    - Entrenamiento con los datos insertados: 
    ``` bash 
    python run.py p01
    ```
    - Predicción con los datos insertados (producción): 
    ``` bash 
    python run.py p02
    ```
    - Análisis de los datos insertados (exploración): 
    ``` bash 
    python run.py p03
    ```

4. Una vez finalizado el proceso revisar la carpeta `files/datasets/output` para visualizar los resultados de predicciones por cliente. **(Resultado del Pipeline 01 y 02)**

5. Examinar las métricas y gráficas de desempeño en las carpeta `files/modeling_output`. **(Resultado del Pipeline 01)**

6. Revisar las visualizaciones generadas en la carpeta `reports/plots/`. **(Resultado del Pipeline 03)**

*Nota: es posible especificar el modelo que se quiere poner a prueba en el archivo pipeline/p01_train_pipeline.py al cambiar la función del paso 3 (3. Modeling creation). Para un modelo de regresión logística usar `create_logistic_regression_model()`, para un random forest usar `create_random_forest_classifier()`  y para un XGBoost usar `create_XGBoost_fixed_model()`. A cada cual se le puede ajustar el umbral de clasificación deseado en el parámetro `threshold`.*

*Siempre es necesario entrenar un modelo (Ejecutar pipeline 01) antes de ponerlo en producción (Ejecutar pipeline 02)*

---

## 🧠 Habilidades técnicas demostradas

- **Análisis exploratorio de datos (EDA)** con visualizaciones.
- **Ingeniería de características** y tratamiento de valores nulos.
- **Balanceo de clases** y manejo de datos desbalanceados.
- **Modelado predictivo** con algoritmos supervisados.
- **Validación cruzada** y métricas de evaluación (Accuracy, Recall, F1, F2, ROC AUC).
- **Estructuración modular** del código en scripts y funciones reutilizables.
- **Pipelines dinámicos** para la organización y ejecución simple de la solución.

## 🛠️ Tecnologías y herramientas utilizadas

- **Python**
- **pandas / NumPy:** manipulación y análisis de datos.
- **scikit-learn:** modelado, validación y métricas.
- **matplotlib / seaborn:** visualización de resultados.
- **Jupyter Notebook:** documentación y flujo de trabajo interactivo.

---

## ✅ Resultados y conclusiones

- Se implementaron tres pipelines de entrenamiento, predicción y análisis con dashboards.

- El mejor modelo (**XGBoost**) alcanzó una exactitud de ~0.83, con un **recall de ~0.92**, al utilizar un umbral de 0.4.

![image](https://raw.githubusercontent.com/AndresCalvete-DataScientist/interconnect-customer-churn-prediction/main/files/modeling_output/figures/XGBClassifier_test.png) 


- Se identificaron las variables más relevantes para explicar la cancelación a través de visualizaciones.

![image](https://raw.githubusercontent.com/AndresCalvete-DataScientist/interconnect-customer-churn-prediction/main/reports/plots/monthly_charge_vs_churn_dashboard.png) 

- Se identificaron segmentos de clientes con alta probabilidad de abandono, lo que permite a Interconnect enfocar sus esfuerzos de retención en estos grupos específicos.

![image](https://raw.githubusercontent.com/AndresCalvete-DataScientist/interconnect-customer-churn-prediction/main/reports/plots/contract_data_dashboard.png) 

## 📄 Jupyter Notebook

El flujo completo del proyecto se encuentra en los notebooks de la carpeta notebooks/, donde se documenta:

- Preprocesamiento de datos.
- Análisis exploratorio.
- Modelado predictivo.
- Evaluación de métricas.
- Conclusiones y resultados.

🔗 Ver notebook completo en GitHub → [interconnect-customer-churn-prediction](https://github.com/AndresCalvete-DataScientist/interconnect-customer-churn-prediction/blob/main/notebooks/eda.ipynb)

---

## 📈 Posibles mejoras futuras

- Desplegar el modelo en una API para predicciones en tiempo real.

---

### 👨‍💻 Autor

**Andrés Calvete**  
Científico de Datos Junior  
[LinkedIn](https://www.linkedin.com/in/andrescalvete/)  
ascalvete@hotmail.com
