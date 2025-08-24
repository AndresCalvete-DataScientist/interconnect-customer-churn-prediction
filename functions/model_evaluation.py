import pandas as pd
import numpy as np
import matplotlib .pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold


def evaluate_model_cv(model, X, y, cv_splits=5, figname=''):    
    # Configuración de la validación cruzada
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=12345)
    metrics_list = []

    # Configuración de los gráficos
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Iterar a través de los pliegues
    for (train_idx, valid_idx) in skf.split(X, y):
        # Dividir los datos
        train_features, valid_features = X[train_idx], X[valid_idx]
        train_target, valid_target = y[train_idx], y[valid_idx]

        # Entrenar el modelo
        model.fit(train_features, train_target)

        # Evaluar el modelo en ambos conjuntos
        eval_stats = {}
        for dataset_type, features, target, color in (
            ('train', train_features, train_target, 'blue'),
            ('valid', valid_features, valid_target, 'green')
        ):
            # Preparar las predicciones
            pred_target = model.predict(features)
            pred_proba = model.predict_proba(features)[:, 1]

            # F1
            f1_thresholds = np.arange(0, 1.01, 0.05)
            f1_scores = [metrics.f1_score(target, pred_proba >= thr) for thr in f1_thresholds]
            
            # ROC
            fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
            roc_auc = metrics.roc_auc_score(target, pred_proba)

            # PRC
            precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred_proba)
            aps = metrics.average_precision_score(target, pred_proba)

            # Plot F1
            ax = axs[0]
            max_f1_score_idx = np.argmax(f1_scores)
            ax.plot(f1_thresholds, f1_scores, color=color, label=f'{dataset_type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
            # establecer cruces para algunos umbrales
            for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
                closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
                if threshold in (0.2, 0.8):         
                    marker_color = 'yellow'
                elif threshold in (0.4, 0.6):       
                    marker_color = 'orange'
                elif threshold == 0.5:              
                    marker_color = 'red'
                ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])
            ax.set_xlabel('threshold')
            ax.set_ylabel('F1')
            ax.legend(loc='lower center')
            ax.set_title(f'Valor F1')
        
            # Plot ROC
            ax = axs[1]
            ax.plot(fpr, tpr, color=color, label=f'{dataset_type}, ROC AUC={roc_auc:.2f}')
            # establecer cruces para algunos umbrales
            for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
                closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
                if threshold in (0.2, 0.8):         
                    marker_color = 'yellow'
                elif threshold in (0.4, 0.6):       
                    marker_color = 'orange'
                elif threshold == 0.5:              
                    marker_color = 'red'
                ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
            ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.legend(loc='lower center')
            ax.set_title(f'Curva ROC')
        
            # Plot PRC
            ax = axs[2]
            ax.plot(recall, precision, color=color, label=f'{dataset_type}, AP={aps:.2f}')
            # establecer cruces para algunos umbrales
            for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
                closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
                if threshold in (0.2, 0.8):         
                    marker_color = 'yellow'
                elif threshold in (0.4, 0.6):       
                    marker_color = 'orange'
                elif threshold == 0.5:              
                    marker_color = 'red'
                ax.plot(recall[closest_value_idx], precision[closest_value_idx], color=marker_color, marker='X', markersize=7)
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])
            ax.set_xlabel('recall')
            ax.set_ylabel('precision')
            ax.legend(loc='lower center')
            ax.set_title(f'PRC')
            
            # Guardar métricas
            eval_stats.setdefault(dataset_type, {})
            eval_stats[dataset_type]['Threshold'] = f1_thresholds[max_f1_score_idx]
            eval_stats[dataset_type]['Accuracy'] = metrics.accuracy_score(target, pred_target)
            eval_stats[dataset_type]['Recall'] = metrics.recall_score(target, pred_target)
            eval_stats[dataset_type]['F1'] = metrics.f1_score(target, pred_target)
            eval_stats[dataset_type]['F2'] = metrics.fbeta_score(target, pred_target, beta=2)
            eval_stats[dataset_type]['APS'] = aps
            eval_stats[dataset_type]['ROC AUC'] = roc_auc

        metrics_list.append(eval_stats)

    print(f"Folds totales: {cv_splits}")
    
    # Promediar métricas de todos los folds
    df_eval_stats = pd.DataFrame(
        {
            'train': pd.DataFrame([m['train'] for m in metrics_list]).mean(),
            'valid': pd.DataFrame([m['valid'] for m in metrics_list]).mean()
        }
    ).round(2)
    
    df_eval_stats = df_eval_stats.reindex(index=('Threshold', 'Accuracy', 'Recall', 'F1', 'F2', 'APS', 'ROC AUC'))

    overfitting = (df_eval_stats['train'].drop('Threshold') - df_eval_stats['valid'].drop('Threshold')).mean()
    
    # Presentar resultados
    print("\nMétricas promedio (validación cruzada):")
    print(df_eval_stats)
    print(f"\nSobreajuste promedio: {overfitting:.2f}")
    
    plt.tight_layout()
    if figname == '':
        fig.savefig(f'files/modeling_output/figures/{type(model).__name__}.png')  
    else:
        fig.savefig(f'files/modeling_output/figures/{figname}.png')

    return df_eval_stats



def evaluate_model_test(model, pred_proba, pred_target, target, threshold_train, figname=''):

    eval_stats = {}

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # F1
    f1_thresholds = np.arange(0, 1.01, 0.05)
    f1_scores = [metrics.f1_score(target, pred_proba>=threshold) for threshold in f1_thresholds]

    # ROC
    fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
    roc_auc = metrics.roc_auc_score(target, pred_proba)
    eval_stats['ROC AUC'] = roc_auc

    # PRC
    precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred_proba)
    aps = metrics.average_precision_score(target, pred_proba)
    eval_stats['APS'] = aps

    color = 'green'

    # Valor F1
    ax = axs[0]
    max_f1_score_idx = np.argmax(f1_scores)
    ax.plot(f1_thresholds, f1_scores, color=color, label=f'test, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
    ax.axvline(threshold_train, color='red', linestyle='--', label='Umbral entrenado')
    # establecer cruces para algunos umbrales
    for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
        closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
        marker_color = 'orange' if threshold != 0.5 else 'red'
        ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('threshold')
    ax.set_ylabel('F1')
    ax.legend(loc='lower center')
    ax.set_title(f'Valor F1')

    # ROC
    ax = axs[1]
    ax.plot(fpr, tpr, color=color, label=f'test, ROC AUC={roc_auc:.2f}')
    ax.axvline(fpr[np.argmin(np.abs(roc_thresholds-threshold_train))], color='red', linestyle='--', label='Umbral entrenado')
    # establecer cruces para algunos umbrales
    for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
        closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
        marker_color = 'orange' if threshold != 0.5 else 'red'
        ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.legend(loc='lower center')
    ax.set_title(f'Curva ROC')

    # PRC
    ax = axs[2]
    ax.plot(recall, precision, color=color, label=f'test, AP={aps:.2f}')
    ax.axvline(recall[np.argmin(np.abs(pr_thresholds-threshold_train))], color='red', linestyle='--', label='Umbral entrenado')
    # establecer cruces para algunos umbrales
    for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
        closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
        marker_color = 'orange' if threshold != 0.5 else 'red'
        ax.plot(recall[closest_value_idx], precision[closest_value_idx], color=marker_color, marker='X', markersize=7)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.legend(loc='lower center')
    ax.set_title(f'PRC')

    eval_stats['Accuracy'] = metrics.accuracy_score(target, pred_target)
    eval_stats['Recall'] = metrics.recall_score(target, pred_target)
    eval_stats['F1'] = metrics.f1_score(target, pred_target)
    eval_stats['F2'] = metrics.fbeta_score(target, pred_target, beta=2)

    df_eval_stats = pd.DataFrame(eval_stats, index=['train']).transpose() # ('Accuracy', 'Recall', 'F1', 'F2', 'APS', 'ROC AUC')
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(['Accuracy', 'Recall', 'F1', 'F2', 'APS', 'ROC AUC'])    

    print(df_eval_stats)
    
    plt.tight_layout()
    if figname == '':
        fig.savefig(f'files/modeling_output/figures/{type(model).__name__}_test.png')  
    else:
        fig.savefig(f'files/modeling_output/figures/{figname}.png')
    
    return df_eval_stats