import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Definir colores personalizados en el mismo orden que las barras (0: No Churn, 1: Churn)
custom_colors = ['#5CB85C', '#D9534F']


def create_target_balance_barplot(contract, filename):
    # Conteo de valores en la columna IsChurn
    churn_counts = contract['IsChurn'].value_counts().sort_index()

    # Visualización de la distribución de la variable IsChurn
    ax = sns.barplot(x=churn_counts.index.astype(str), y=churn_counts.values,
                     hue=churn_counts.index.astype(str), palette=custom_colors)

    # Etiquetas y título
    for i, value in enumerate(churn_counts.values):
        ax.text(i, value + max(churn_counts.values) * 0.001,
                f'{value:,}', ha='center', va='bottom')

    plt.xlabel('')
    plt.ylabel('Cantidad de clientes')
    plt.title('Distribución de la variable objetivo', pad=20)
    plt.xticks([0, 1], ['Permanecen', 'Abandonaron'])
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(f'reports/plots/{filename}.png')


def create_churn_by_column_barplot(dataset, column, title, xlabel, filename, percentage=False):
    # Agrupar datos por columna y churn, y organizarlos
    type_vs_churn = dataset.groupby(
        column)['IsChurn'].value_counts().unstack()

    # Visualización de la distribución de la columna en funcion del abandono
    ax = type_vs_churn.plot(kind='bar', stacked=True, color=custom_colors)

    # Añadir etiquetas de porcentaje
    if percentage:
        type_vs_churn_percentage = dataset.groupby(
            column)['IsChurn'].value_counts(normalize=True).unstack().round(2)

        for i, array in enumerate(type_vs_churn.values):
            for j, value in enumerate(array):
                ax.text(i + 0.34, np.cumsum(array)[
                        j], f'{100*type_vs_churn_percentage.values[i, j]:.0f}%', ha='center', va='bottom', weight='bold')

    # Etiquetas y título
    plt.xlabel(xlabel)
    plt.ylabel('Cantidad de clientes')
    plt.title(title, pad=20)
    plt.legend(title='Estado del cliente', labels=['Permanece', 'Abandonó'])
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(f'reports/plots/{filename}.png')


def create_churn_by_charge_distribution_dashboard(dataset, column, charge_name, title, filename):
    # Visualización de la distribución de los cargos en funcion del abandono
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histograma en el primer subplot
    sns.histplot(
        data=dataset, x=column, hue='IsChurn', bins=30,
        kde=True, palette=custom_colors, ax=axes[0], element='step', stat='density', legend=True
    )
    axes[0].set_xlabel(charge_name)
    axes[0].set_ylabel('Densidad')
    axes[0].legend(title='Estado del cliente', labels=[
                   'Abandono', 'KDE-Abandono', 'Permanencia', 'KDE-Permanencia'])
    axes[0].grid(axis='both', alpha=0.5)

    # Boxplot en el segundo subplot
    sns.boxplot(
        data=dataset, x='IsChurn', y=column, hue='IsChurn', palette=custom_colors, ax=axes[1], legend=False
    )
    axes[1].set_xlabel('')
    axes[1].set_ylabel(charge_name)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Permanencia', 'Abandono'])

    axes[1].grid(axis='y', alpha=0.5)

    # Título
    fig.suptitle(title, fontsize=16, y=1)
    plt.tight_layout()
    plt.savefig(f'reports/plots/{filename}.png')


def difference():
    # Calcula la duración en meses antes del abandono
    def diff_months(row):
        if pd.notnull(row['EndDate']) and pd.notnull(row['BeginDate']):
            end = row['EndDate']
            begin = row['BeginDate']
            return (end.year - begin.year) * 12 + (end.month - begin.month)
        else:
            return np.nan

    # Aplica la función para calcular la duración en meses
    df_contract['DurationMonths'] = df_contract.apply(diff_months, axis=1)
    df_contract['DurationMonths'].describe()

    # Visualización de la duración de clientes en meses antes del abandono
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histograma en el primer subplot
    sns.histplot(
        data=df_contract, x='DurationMonths', color=custom_colors[1], bins=30,
        kde=True, ax=axes[0], element='step', legend=True
    )
    axes[0].set_xlabel('Meses de permanencia antes del abandono')
    axes[0].set_ylabel('Cantidad de clientes')
    axes[0].grid(axis='both', alpha=0.5)

    # Boxplot en el segundo subplot
    sns.boxplot(
        data=df_contract, x='DurationMonths', color=custom_colors[1], ax=axes[1], legend=False, orient='horizontal'
    )
    axes[1].set_ylabel('')
    axes[1].set_xlabel('Meses de permanencia antes del abandono')
    axes[1].grid(axis='y', alpha=0.5)

    # Título
    fig.suptitle(
        'Hay mayor riesgo de abandono durante los primeros meses', fontsize=16, y=1)
    plt.tight_layout()
    plt.show()
