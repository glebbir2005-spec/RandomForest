from sklearn.model_selection import train_test_split
from data_preprocessing import DataPreprocessor
from experiment import run_experiments
from visualization import plot_results

if __name__ == "__main__":
    data_paths = ['adult_data.csv', 'adult_test.csv']
    target_col = 'income'
    
    preprocessor = DataPreprocessor(data_paths, target_col)
    X, y, feature_names, class_names = preprocessor.load_and_preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=67, stratify=y
    )

    estimators_range = range(50, 101, 10)
    
    experiment_results = run_experiments(
        X_train, y_train, X_test, y_test, 
        estimators_range=estimators_range, 
    )


# GINI Размер train: 90% | Accuracy: 0.8282 | Precision: 0.6595 | Recall: 0.6693 | F1: 0.6644
    dt_previous_metrics = {
        'Accuracy': 0.8282,
        'Precision': 0.6595,
        'Recall': 0.6693,
        'F1-score': 0.6644
    }

    print("Построение графиков...")
    plot_results(experiment_results, dt_metrics=dt_previous_metrics)