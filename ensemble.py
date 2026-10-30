from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class EnsembleClassifier:
    def __init__(self, n_estimators=100, **kwargs):
        """
        :param technique: Техника ансамблирования
        :param n_estimators: Количество участников ансамбля
        :param kwargs: Дополнительные параметры для выбранной техники
        """
        self.n_estimators = n_estimators
        self.kwargs = kwargs
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, **self.kwargs)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
            'F1-score': f1_score(y_test, y_pred, average='binary', zero_division=0)
        }
        return metrics