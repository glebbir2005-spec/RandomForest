import pandas as pd

class DataPreprocessor:
    def __init__(self, data_paths, target_col):
        """
        :param data_paths: список путей к файлам (например, ['train.csv', 'test.csv'])
        :param target_col: название целевой переменной
        """
        self.data_paths = data_paths if isinstance(data_paths, list) else [data_paths]
        self.target_col = target_col
        self.class_names = ['<=50K', '>50K']

    def load_and_preprocess(self):
        print("Загрузка и объединение данных...")
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                   'marital-status', 'occupation', 'relationship', 'race', 'sex',
                   'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
        
        df_list = []
        for path in self.data_paths:
            temp_df = pd.read_csv(path, names=columns, na_values="?", skipinitialspace=True, comment='|')
            df_list.append(temp_df)
            
        df = pd.concat(df_list, ignore_index=True)
        
        df.dropna(inplace=True)
        
        X_raw = df.drop(self.target_col, axis=1)
        y_raw = df[self.target_col]
        
        y_raw = y_raw.astype(str).str.replace('.', '', regex=False)
        y = y_raw.map({'<=50K': 0, '>50K': 1})
        
        X = pd.get_dummies(X_raw)
        feature_names = X.columns.tolist()
        
        print(f"Данные загружены. Размерность объединенного датасета: {X.shape[0]} строк, {X.shape[1]} признаков.\n")
        return X, y, feature_names, self.class_names