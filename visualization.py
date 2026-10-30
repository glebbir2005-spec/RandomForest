import matplotlib.pyplot as plt

def plot_results(experiment_results, dt_metrics=None):
    """
    Отрисовывает графики показателей качества классификации.
    :param experiment_results: Список словарей с метриками ансамблей.
    :param dt_metrics: Словарь с метриками одиночного дерева.
    """
    n_estimators = [res['n_estimators'] for res in experiment_results]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    
    plt.figure(figsize=(14, 10))
    
    for i, metric in enumerate(metrics_names, 1):
        plt.subplot(2, 2, i)
        
        rf_values = [res[metric] for res in experiment_results]
        plt.plot(n_estimators, rf_values, marker='o', label=f'Случайный лес ({metric})', color='b')
        
        if dt_metrics and metric in dt_metrics:
            dt_val = dt_metrics[metric]
            plt.axhline(y=dt_val, color='r', linestyle='--', label=f'Дерево решений')
        
        plt.title(f'Зависимость {metric} от числа участников')
        plt.xlabel('Количество деревьев в ансамбле')
        plt.ylabel(metric)
        plt.xticks(n_estimators)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    plt.show()