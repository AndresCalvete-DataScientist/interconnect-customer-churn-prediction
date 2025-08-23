from functions.exploratory_plots import create_churn_by_column_barplot
import pandas as pd
import matplotlib.pyplot  as plt


def plot():


    # 1. Loading data ----------------------------------------  


    df_contract = pd.read_csv("files/datasets/intermediate/a01_contract_cleaned.csv", parse_dates=['BeginDate', 'EndDate'])
    df_internet = pd.read_csv("files/datasets/intermediate/a01_internet_cleaned.csv")


    # 2. Merge dataframes ----------------------------------------  
    
    
    df_con_int = df_contract.merge(df_internet, how='left', on='customerID', validate='1:1')
    
    services=['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_con_int[services] = df_con_int[services].fillna(0)

    
    # 3. Plotting and save ----------------------------------------  


    # Has internet barplot
    create_churn_by_column_barplot(
        df_con_int, 'InternetService', '',
        title='Servicio de Internet en función del abandono',
        filename='internet_service_vs_churn',
        percentage=True,  
        xticks=['DSL', 'Fibra óptica', 'Ninguno'],  
        dropna=False)
    
    # Service bar plots dashboard
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    
    # Seguridad en línea
    create_churn_by_column_barplot(
        df_con_int, 'OnlineSecurity', 'Servicio de seguridad en línea',
        title='Servicio de seguridad en línea en función del abandono',
        percentage=True,    
        xticks=['Inactivo', 'Activo'],    
        output='return', ax=axes[0, 0])
    
    # Respaldo en línea
    create_churn_by_column_barplot(
        df_con_int, 'OnlineBackup', 'Servicio de respaldo en línea',
        title='Servicio de respaldo en línea en función del abandono',
        percentage=True,    
        xticks=['Inactivo', 'Activo'],    
        output='return', ax=axes[0, 1])
    
    # Protección de dispositivo
    create_churn_by_column_barplot(
        df_con_int, 'DeviceProtection', 'Servicio de protección de dispositivo',
        title='Servicio de protección de dispositivo en función del abandono',
        percentage=True,    
        xticks=['Inactivo', 'Activo'],    
        output='return', ax=axes[1, 0])
    
    # Soporte técnico
    create_churn_by_column_barplot(
        df_con_int, 'TechSupport', 'Servicio de soporte técnico',
        title='Servicio de soporte técnico en función del abandono',
        percentage=True,    
        xticks=['Inactivo', 'Activo'],    
        output='return', ax=axes[1, 1])
    
    # Streaming de TV
    create_churn_by_column_barplot(
        df_con_int, 'StreamingTV', 'Servicio de streaming de TV',
        title='Servicio de streaming de TV en función del abandono',
        percentage=True,    
        xticks=['Inactivo', 'Activo'],    
        output='return', ax=axes[2, 0])
    
    # Streaming de películas
    create_churn_by_column_barplot(
        df_con_int, 'StreamingMovies', 'Servicio de streaming de películas',
        title='Servicio de streaming de películas en función del abandono',
        percentage=True,    
        xticks=['Inactivo', 'Activo'],    
        output='return', ax=axes[2, 1])
    
    
    plt.tight_layout()
    fig.savefig(f'reports/plots/internet_data_dashboard.png')
    
    print("📊 Visualizaciones de servicios de internet creadas.")
    
    
if __name__ == "__main__":
    plot()