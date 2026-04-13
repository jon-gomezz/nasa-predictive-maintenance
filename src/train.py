import sys
from pathlib import Path

# Añadimos el directorio raíz al path para que Python encuentre 'src'
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data import load_cmapss_data, get_train_val_split
from src.features import add_binary_target, drop_constant_columns, add_rolling_features
from src.models import train_xgboost_baseline
from sklearn.metrics import classification_report, confusion_matrix

def main():
    print("1. Cargando datos originales (train_FD001.txt)...")
    df = load_cmapss_data('train_FD001.txt')
    
    print("2. Procesando variables, generando target (30 ciclos) y feature engineering...")
    df = add_binary_target(df, window_size=30)
    df = drop_constant_columns(df)
    
    # NUEVA FASE 2: Añadimos memoria temporal
    print("   -> Calculando ventanas deslizantes (Rolling features)...")
    df = add_rolling_features(df, windows=[5, 15])
    
    # Rellenamos los NaNs generados por la std en el ciclo 1 (donde no hay 2 datos) con 0
    df.fillna(0, inplace=True)
    
    
    target_col = 'failure_within_30_cycles'
    
    print("3. Ejecutando Split protegido por Motor (unit_id)...")
    train_df, val_df = get_train_val_split(df, val_size=0.2)
    
    # MUY IMPORTANTE: Evitar Data Leakage.
    # No podemos pasarle al modelo la Vida Útil real (rul) porque el modelo 
    # estaría mirando la respuesta. Tampoco le pasamos el Target ni el unit_id.
    # Sí le pasamos el time_cycle, ya que en la vida real sabremos cuánto tiempo 
    # lleva encendido el motor de nuestro cliente.
    cols_to_drop = ['unit_id', 'rul', target_col]
    
    X_train = train_df.drop(columns=cols_to_drop)
    y_train = train_df[target_col]
    
    X_val = val_df.drop(columns=cols_to_drop)
    y_val = val_df[target_col]
    
    print(f"   -> Motores en Train: {train_df['unit_id'].nunique()} | Ciclos (filas): {len(X_train)}")
    print(f"   -> Motores en Val:   {val_df['unit_id'].nunique()} | Ciclos (filas): {len(X_val)}")
    
    print("\n4. Entrenando Baseline XGBoost Classifier...")
    model = train_xgboost_baseline(X_train, y_train)
    
    print("5. Evaluando en el set de Validación (Motores nunca antes vistos)...")
    y_pred = model.predict(X_val)
    
    print("\n================== MATRIZ DE CONFUSIÓN ==================")
    print("                  Predicho: SANO (0)   Predicho: FALLO (1)")
    cm = confusion_matrix(y_val, y_pred)
    print(f"Real: SANO (0)  |        {cm[0][0]}        |         {cm[0][1]}          |")
    print(f"Real: FALLO (1) |        {cm[1][0]}        |         {cm[1][1]}          |")
    
    print("\n================ REPORTE DE CLASIFICACIÓN ===============")
    print(classification_report(y_val, y_pred, target_names=['SANO (0)', 'FALLO (1)']))
    
    import os
    import joblib
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_baseline.joblib')
    print("\n   -> Modelo guardado exitosamente en 'models/xgboost_baseline.joblib'")
    
    print("\n¡Pipeline de Fase 1 (MVP) Ejecutado con Éxito!")

if __name__ == "__main__":
    main()
