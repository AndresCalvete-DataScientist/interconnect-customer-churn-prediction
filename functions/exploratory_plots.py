from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


def create_churn_by_column_barplot(dataset, column, xlabel, title='', filename='', percentage=False, output: Literal['save', 'return', 'both'] = 'save'):
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

    if output == 'save' | output == 'both':
        if filename == '':
            plt.savefig(f'reports/plots/churn_vs_{column}.png')
        else:
            plt.savefig(f'reports/plots/{filename}.png')

    if output == 'return' | output == 'both':
        return ax


def create_churn_by_column_distribution_dashboard(dataset, column, xlabel, title, filename):
    # Visualización de la distribución de los cargos en funcion del abandono
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histograma en el primer subplot
    sns.histplot(
        data=dataset, x=column, hue='IsChurn', bins=30,
        kde=True, palette=custom_colors, ax=axes[0], element='step', stat='density', legend=True
    )
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('Densidad')
    axes[0].legend(title='Estado del cliente', labels=[
                   'Abandono', 'KDE-Abandono', 'Permanencia', 'KDE-Permanencia'])
    axes[0].grid(axis='both', alpha=0.5)

    # Boxplot en el segundo subplot
    sns.boxplot(
        data=dataset, x='IsChurn', y=column, hue='IsChurn', palette=custom_colors, ax=axes[1], legend=False
    )
    axes[1].set_xlabel('')
    axes[1].set_ylabel(xlabel)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Permanencia', 'Abandono'])

    axes[1].grid(axis='y', alpha=0.5)

    # Título
    fig.suptitle(title, fontsize=16, y=1)
    plt.tight_layout()
    plt.savefig(f'reports/plots/{filename}.png')


def create_time_before_churn_distribution_dashboard(dataset, filename):
    # Calcula la duración en meses antes del abandono
    def diff_months(row):
        if pd.notnull(row['EndDate']) and pd.notnull(row['BeginDate']):
            end = row['EndDate']
            begin = row['BeginDate']
            return (end.year - begin.year) * 12 + (end.month - begin.month)
        else:
            return np.nan

    # Aplica la función para calcular la duración en meses
    dataset['DurationMonths'] = dataset.apply(diff_months, axis=1)

    create_churn_by_column_distribution_dashboard(
        dataset, 'DurationMonths', 'Meses de permanencia antes del abandono', 'Permanencia de clientes antes del abandono', filename)


def create_clients_events_timeline_dashboard(dataset, filename):
    # Visualización de la evolución en el abandono y subscripción de clientes mensual.
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Diagrama de lineas de la evolución de la llegada de clientes mensual
    sns.lineplot(
        data=dataset['BeginDate'].value_counts().sort_index(), color=custom_colors[0], ax=axes[0],  linewidth=2
    )
    axes[0].set_title(
        'Evolución de la llegada de clientes mensual', fontsize=14, pad=20)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Nuevas subscribciones')
    axes[0].grid(axis='both', alpha=0.5)

    # Diagrama de lineas de la evolución del abandono mensual
    sns.lineplot(
        data=dataset['EndDate'].value_counts(), color=custom_colors[1], ax=axes[1], linewidth=2
    )
    axes[1].set_title(
        'Evolución del abandono mensual', fontsize=14, pad=20)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Contratos terminados')
    axes[1].grid(axis='both', alpha=0.5)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    # Compilación de las subscripciones y abandonos
    sns.lineplot(
        data=dataset['BeginDate'].value_counts().sort_index(), color=custom_colors[0], ax=axes[2],  linewidth=2, label='Subscripciones'
    )
    sns.lineplot(
        data=dataset['EndDate'].value_counts(), color=custom_colors[1], ax=axes[2], linewidth=2, label='Abandonos'
    )
    axes[2].set_title(
        'Compilación de las subscripciones y abandonos', fontsize=14, pad=20)
    axes[2].set_xlabel('')
    axes[2].set_ylabel('Número de eventos')
    axes[2].grid(axis='both', alpha=0.5)
    axes[2].legend(title='Comportamiento')

    plt.tight_layout()
    plt.savefig(f'reports/plots/{filename}.png')


def create_client_flow_lineplot(dataset, filename):
    # Flujo de clientes mensualmente y clientes totales
    clients_flow = pd.DataFrame(
        dataset['BeginDate'].value_counts().sort_index())
    clients_flow = clients_flow.merge(dataset['EndDate'].value_counts(
    ), how='left', left_index=True, right_index=True, suffixes=['_subscribe', '_churn'])
    clients_flow['count_churn'] = clients_flow['count_churn'].fillna(0)
    clients_flow['total_clients'] = clients_flow['count_subscribe'].cumsum(
    ) - clients_flow['count_churn'].cumsum()

    # Visualización del crecimiento del negocio en función del número de clientes
    fig, axes = plt.subplots(figsize=(14, 6))

    # Histograma en el segundo subplot
    sns.lineplot(
        data=clients_flow, x=clients_flow.index, y='total_clients', color=custom_colors[0], linewidth=2
    )

    axes.set_title(
        'Crecimiento del negocio en función del número de clientes', fontsize=14, pad=20)
    axes.set_xlabel('')
    axes.set_ylabel('Total de clientes')
    axes.grid(axis='both', alpha=0.5)

    plt.savefig(f'reports/plots/{filename}.png')


def create_churn_by_year_month_barplots(dataset, filename):
    # Agrupar datos por año y mes de subscripción y churn, y organizarlos
    month_vs_churn = dataset.groupby(
        'BeginMonth')['IsChurn'].value_counts().unstack()
    month_vs_churn_percentage = dataset.groupby(
        'BeginMonth')['IsChurn'].value_counts(normalize=True).unstack().round(2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0] = create_churn_by_column_barplot(
        dataset, 'BeginYear', 'Año de subscripción', 'Año de suscripción en funcion del abandono', output='return')

    # Visualización de la distribución de la variable BeginMonth en funcion del abandono
    month_vs_churn.plot(kind='bar', stacked=True,
                        color=custom_colors, ax=axes[1], rot=0, legend=False)

    # Etiquetas y título
    for i, array in enumerate(month_vs_churn.values):
        for j, value in enumerate(array):
            axes[1].text(i, np.cumsum(array)[
                         j], f'{100*month_vs_churn_percentage.values[i, j]:.0f}%', ha='center', va='bottom', fontsize=7)

    axes[1].set_xlabel('Mes de subscripción')
    axes[1].set_ylabel('Clientes suscritos')
    axes[1].set_title('Mes de suscripción en funcion del abandono', pad=20)
    axes[1].grid(axis='y', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'reports/plots/{filename}.png')
