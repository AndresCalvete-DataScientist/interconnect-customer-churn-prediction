from functions.exploratory_plots import create_churn_by_column_barplot
import pandas as pd
import matplotlib.pyplot  as plt


def plot():


    # 1. Loading data ----------------------------------------  


    df_contract = pd.read_csv("files/datasets/intermediate/a01_contract_cleaned.csv", parse_dates=['BeginDate', 'EndDate'])
    df_personal = pd.read_csv("files/datasets/intermediate/a01_personal_cleaned.csv")


    # 2. Merge dataframes ----------------------------------------  
    
    
    df_con_per = df_contract.merge(df_personal, how='inner', on='customerID', validate='1:1')

    
    # 3. Plotting and save ----------------------------------------  


    # Crear bar plots dashboard
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Género
    create_churn_by_column_barplot(
        df_con_per, 'gender', '',
        title='Género en función del abandono',
        percentage=True,        
        output='return', ax=axes[0, 0])
    
    # Seniority
    create_churn_by_column_barplot(
        df_con_per, 'SeniorCitizen', '',
        title='Seniority en función del abandono',
        percentage=True, 
        xticks=['No Senior', 'Senior'],   
        output='return', ax=axes[0, 1])
    
    # Soltería
    create_churn_by_column_barplot(
        df_con_per, 'Partner', '',
        title='Soltería en función del abandono',
        percentage=True, 
        xticks=['Soltero/a', 'Con pareja'],       
        output='return', ax=axes[1, 0])
    
    axes[1, 0].legend(
    title='Estado del cliente', 
    labels=['Permanece', 'Abandonó'], 
    bbox_to_anchor=(0.02, 0.18), 
    loc='upper left',
    borderaxespad=0.)
    
    # Dependientes
    create_churn_by_column_barplot(
        df_con_per, 'Dependents', '',
        title='Dependientes en función del abandono',
        percentage=True,
        xticks=['Sin dependientes', 'Con dependientes'],         
        output='return', ax=axes[1, 1])    
    
    plt.tight_layout()
    fig.savefig(f'reports/plots/personal_data_dashboard.png')
    
    print("📊 Visualizaciones de datos personales creadas.")
    
    
if __name__ == "__main__":
    plot()