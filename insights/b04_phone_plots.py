from functions.exploratory_plots import create_churn_by_column_barplot
import pandas as pd


def plot():


    # 1. Loading data ----------------------------------------  


    df_contract = pd.read_csv("files/datasets/intermediate/a01_contract_cleaned.csv", parse_dates=['BeginDate', 'EndDate'])
    df_phone = pd.read_csv("files/datasets/intermediate/a01_phone_cleaned.csv")


    # 2. Merge dataframes ----------------------------------------  
    
    
    df_con_pho = df_contract.merge(df_phone, how='left', on='customerID', validate='1:1')
    
    
    # 3. Plotting and save ----------------------------------------  


    # Phone Lines barplot
    create_churn_by_column_barplot(
        df_con_pho, 'MultipleLines', '',
        title='Servicio de Telefonía en función del abandono',
        filename='phone_service_vs_churn',
        percentage=True,  
        xticks=['Línea única', 'Múltiples líneas', 'Sin servicio'],  
        dropna=False)
    
    print("📊 Visualizaciones de servicios de telefónicos creadas.")
    
    
if __name__ == "__main__":
    plot()