from ensemble import EnsembleClassifier

def run_experiments(X_train, y_train, X_test, y_test, estimators_range):
    """
    Проводит серию экспериментов, варьируя количество участников ансамбля.
    """
    results = []
    print(f"\nЗапуск экспериментов")
    
    for n in estimators_range:
        print(f"Обучение леса с {n} деревьями...")
        classifier = EnsembleClassifier(n_estimators=n, random_state=42)
        classifier.train(X_train, y_train)
        metrics = classifier.evaluate(X_test, y_test)
        
        metrics['n_estimators'] = n
        results.append(metrics)
        
    print("Эксперименты завершены.\n")
    return results