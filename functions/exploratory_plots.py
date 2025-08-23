from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

# Definir colores personalizados en el mismo orden que las barras (0: No Churn, 1: Churn)
custom_colors = ['#5CB85C', '#D9534F']


def create_target_balance_barplot(contract, filename='target_balance'):
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



def create_churn_by_column_barplot(dataset, column, xlabel, title='', xticks=None,
                                   filename='', rot=0, percentage=False, output='save', ax=None, fontsize=10, dropna=True):
    # Agrupar datos por columna y churn
    type_vs_churn = dataset.groupby(column, dropna=dropna)['IsChurn'].value_counts().unstack()
    # if not dropna:
    #     print(len(type_vs_churn.index))
    #     print(dataset[dataset[column].isna()])
    # Crear figura solo si no se pasa ax
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure  # Para poder guardar luego si es necesario

    # Plot
    type_vs_churn.plot(kind='bar', stacked=True, color=custom_colors, ax=ax, rot=rot)
            
    # Porcentajes
    if percentage:
        type_vs_churn_percentage = dataset.groupby(column, dropna=dropna)['IsChurn'].value_counts(normalize=True).unstack().round(2)
        for i, array in enumerate(type_vs_churn.values):
            for j, value in enumerate(array):
                if value > np.sum(type_vs_churn, axis=1).max()*0.05:
                    ax.text(i, np.cumsum(array)[j] - (value/2),
                            f'{100*type_vs_churn_percentage.values[i, j]:.0f}%',
                            ha='center', va='center', fontsize=fontsize, color='white', fontweight='bold')

    # Etiquetas y título
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cantidad de clientes')
    ax.set_title(title, pad=20)
    ax.legend(title='Estado del cliente', labels=['Permanece', 'Abandonó'])
    ax.grid(axis='y', alpha=0.5)
    
    # Cambiar etiquetas del eje x si se pasa xticks
    if xticks is not None:
        if len(xticks) != len(type_vs_churn.index):
            raise ValueError("La longitud de xticks no coincide con la cantidad de categorías")
        ax.set_xticklabels(xticks)

    plt.tight_layout()

    # Guardar figura solo si se creó figura nueva o se indica output
    if output in ['save', 'both']:
        if filename == '':
            fig.savefig(f'reports/plots/churn_vs_{column}.png')
        else:
            fig.savefig(f'reports/plots/{filename}.png')

    return ax



def create_churn_by_column_distribution_dashboard(dataset, column, xlabel, title, filename=''):
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
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()    
    
    if filename == '':
        plt.savefig(f'reports/plots/churn_vs_{column}.png')
    else:
        plt.savefig(f'reports/plots/{filename}.png')



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
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Año de suscripción
    create_churn_by_column_barplot(
        dataset, 'BeginYear', 'Año de subscripción',
        title='Año de suscripción en función del abandono',
        percentage=True,
        fontsize=9,
        output='return', ax=axes[0])
    
    # Mes de suscripción
    create_churn_by_column_barplot(
        dataset, 'BeginMonth', 'Mes de subscripción',
        title='Mes de suscripción en función del abandono',
        percentage=True,
        fontsize=7,
        output='return', ax=axes[1])

    plt.tight_layout()
    fig.savefig(f'reports/plots/{filename}.png')

