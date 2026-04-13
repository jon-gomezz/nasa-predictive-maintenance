import os
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data import load_cmapss_data, get_train_val_split
from src.features import add_binary_target, drop_constant_columns, add_rolling_features, ConditionNormalizer

def data_pipeline():
    print("1. Cargando datos de régimen múltiple (train_FD002.txt)...")
    df = load_cmapss_data('train_FD002.txt')
    
    print("2. Procesando variables objetivo (30 ciclos)...")
    df = add_binary_target(df, window_size=30)
    
    # 3. Split Seguro
    train_df, val_df = get_train_val_split(df, val_size=0.2)
    
    # 4. K-MEANS & NORMALIZACIÓN MULTI-RÉGIMEN (El gran cambio)
    print("4. Detectando Regímenes Ocultos (K-Means) y Escalandos Sensores aisildamente...")
    normalizer = ConditionNormalizer(n_clusters=6)
    train_df = normalizer.fit_transform(train_df)
    val_df = normalizer.transform(val_df)
    
    # 5. Ventanas Retrospectivas (Rolling) sobre señales ESTABILIZADAS
    print("5. Calculando tendencias temporales sobre sensores normalizados...")
    train_df = add_rolling_features(train_df, windows=[5, 15])
    val_df = add_rolling_features(val_df, windows=[5, 15])
    
    # 6. Limpieza de Ruido
    train_df.fillna(0, inplace=True)
    val_df.fillna(0, inplace=True)
    
    train_df = drop_constant_columns(train_df)
    # Match validation columns
    val_df = val_df[train_df.columns]
    
    cols_to_drop = ['unit_id', 'time_cycle', 'rul', 'failure_within_30_cycles']
    X_train = train_df.drop(columns=cols_to_drop)
    y_train = train_df['failure_within_30_cycles']
    X_val = val_df.drop(columns=cols_to_drop)
    y_val = val_df['failure_within_30_cycles']
    
    # Guardamos el pipeline de normalización vital para producción
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(normalizer, 'models/condition_normalizer_fd002.joblib')
    
    return X_train, y_train, X_val, y_val

def build_xgb_objective(X_train, y_train, X_val, y_val):
    ratio = float(y_train.value_counts()[0] / y_train.value_counts()[1])
    
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'scale_pos_weight': ratio * trial.suggest_float('scale_pos_multiplier', 0.8, 1.5),
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train
        model = xgb.XGBClassifier(**param)
        model.fit(X_train, y_train)
        
        # Predict on validation blind set
        y_pred = model.predict(X_val)
        
        # Queremos maximizar explícitamente el Recall penalizando un poco el Accuracy si es necesario
        # Usaremos el F1 ponderado, pero vamos a priorizar la F1 de la clase 1 (Fallo)
        score = f1_score(y_val, y_pred, average='binary') 
        return score

    return objective

def main():
    X_train, y_train, X_val, y_val = data_pipeline()
    
    print("\n--- INICIANDO OPTUNA HYPERPARAMETER TUNING ---")
    study = optuna.create_study(direction='maximize')
    objective = build_xgb_objective(X_train, y_train, X_val, y_val)
    
    # Ejecutamos 15 intentos para no demorar (en un pipeline real serían 100)
    study.optimize(objective, n_trials=15)
    
    print("\nMejores hiperparámetros encontrados por Optuna:")
    print(study.best_params)
    
    print("\nEntrenando Modelo Final FD002...")
    best_params = study.best_params
    base_ratio = float(y_train.value_counts()[0] / y_train.value_counts()[1])
    best_params['scale_pos_weight'] = base_ratio * best_params.pop('scale_pos_multiplier')
    best_params['use_label_encoder'] = False
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = 42
    
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)
    
    y_pred = final_model.predict(X_val)
    
    print("\n================== MATRIZ DE CONFUSIÓN MULTI-RÉGIMEN (FD002) ==================")
    cm = confusion_matrix(y_val, y_pred)
    print("                  Predicho: SANO (0)   Predicho: FALLO (1)")
    print(f"Real: SANO (0)  |        {cm[0][0]:<4}        |         {cm[0][1]:<4}          |")
    print(f"Real: FALLO (1) |        {cm[1][0]:<4}        |         {cm[1][1]:<4}          |")
    
    print("\n================ REPORTE DE CLASIFICACIÓN ===============")
    print(classification_report(y_val, y_pred, target_names=['SANO (0)', 'FALLO (1)']))
    
    joblib.dump(final_model, 'models/xgboost_fd002_tuned.joblib')
    print("-> Modelo robusto (Optuna + KMeans) guardado en 'models/xgboost_fd002_tuned.joblib'")

if __name__ == "__main__":
    main()
