import xgboost as xgb
import pandas as pd

def train_xgboost_baseline(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """
    Instancia y entrena un modelo XGBoost base (MVP) para clasificación binaria.
    Utilizamos los parámetros principales por defecto para fijar nuestra línea base.
    """
    # Calculamos el ratio de desbalanceo para penalizar severamente los Falsos Negativos
    # scale_pos_weight = count(negative instances) / count(positive instances)
    num_negatives = (y_train == 0).sum()
    num_positives = (y_train == 1).sum()
    weight_ratio = num_negatives / num_positives
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=random_state,
        eval_metric='logloss',
        scale_pos_weight=weight_ratio
    )
    
    model.fit(X_train, y_train)
    return model
