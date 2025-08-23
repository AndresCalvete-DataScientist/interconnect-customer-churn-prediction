from functions.exploratory_plots import *
import pandas as pd


def plot():


    # 1. Loading data ----------------------------------------  


    df_contract = pd.read_csv("files/datasets/intermediate/a01_contract_cleaned.csv", parse_dates=['BeginDate', 'EndDate'])


    # 2. Plotting and save ----------------------------------------  


    # Crear plot de balance del objetivo
    create_target_balance_barplot(df_contract)
    
   
    # Crear barplots dashboard
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))    
    
    # Tipo de contrato
    create_churn_by_column_barplot(
        df_contract, 'Type', '',
        title='Tipo de contrato en función del abandono',         
        rot=45, 
        percentage=True, 
        output='return', ax=axes[0])
    
    # Método de pago
    create_churn_by_column_barplot(
        df_contract, 'PaymentMethod', '',
        title='Método de pago en función del abandono',         
        rot=45, 
        percentage=True, 
        output='return', ax=axes[1])
    
    # Tipo de facturación
    create_churn_by_column_barplot(
        df_contract, 'PaperlessBilling', '',
        title='Tipo de facturación en función del abandono',         
        rot=45, 
        xticks=['Normal', 'Electrónica'],
        percentage=True, 
        output='return', ax=axes[2])

    plt.tight_layout()
    fig.savefig(f'reports/plots/contract_data_dashboard.png')
    
    # Año y mes de abandono
    create_churn_by_year_month_barplots(
        df_contract, filename='year_and_month_vs_churn')
    

    # Crear charge dashboards
    create_churn_by_column_distribution_dashboard(
        df_contract, 'MonthlyCharges', 'Cargos mensuales', 'Cargos mensuales en función del abandono', filename='monthly_charge_vs_churn_dashboard')

    create_churn_by_column_distribution_dashboard(
        df_contract, 'TotalCharges', 'Cargos totales', 'Cargos totales en función del abandono', filename='total_charge_vs_churn_dashboard')

    create_churn_by_column_distribution_dashboard(
        df_contract, 'DurationMonths', 'Meses de permanencia antes del abandono', 'Permanencia de clientes antes del abandono', filename='before_churn_dashboard')
    

    # Crear line plots
    create_clients_events_timeline_dashboard(
        df_contract, filename='client_events_timeline_dashboard')

    create_client_flow_lineplot(df_contract, filename='clients_flow')
    
    print("📊 Visualizaciones de datos de contratos creadas.")


if __name__ == "__main__":
    plot()