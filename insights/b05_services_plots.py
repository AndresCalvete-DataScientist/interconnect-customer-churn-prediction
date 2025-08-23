from functions.exploratory_plots import create_churn_by_column_barplot
import pandas as pd


def plot():


    # 1. Loading data ----------------------------------------  


    data = pd.read_csv("files/datasets/intermediate/a01_data_cleaned.csv")
    

    # 2. Plotting and save ----------------------------------------  


    # All Services barplot
    create_churn_by_column_barplot(
        data, 'AllServicesCount', 'Número de servicios adquiridos',
        title='Número de servicios adquiridos en función del abandono',
        filename='all_services_vs_churn',
        percentage=True,  
    )
    
    print("📊 Visualizaciones de todos los servicios creadas.")
    
    
if __name__ == "__main__":
    plot()