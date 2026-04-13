import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data import load_cmapss_data
from src.features import add_rolling_features, ConditionNormalizer

def main():
    print("====== EXAMEN FINAL MULTI-RÉGIMEN: SET DE TEST CIEGO (FD002) ======\n")
    
    # 1. Cargar el histórico en vivo (test_FD002.txt)
    print("1. Cargando telemetría de turbinas activas bajo perturbaciones climáticas severas...")
    df_test = load_cmapss_data('test_FD002.txt')
    
    # 2. Despertar el normalizador (Vital para FD002)
    print("2. Despertando el pipeline de normalización (K-Means)...")
    normalizer = joblib.load('models/condition_normalizer_fd002.joblib')
    df_test = normalizer.transform(df_test)
    
    # Añadimos memoria temporal tras aislar el ruido de los regímenes
    print("   -> Calculando ventanas deslizantes sobre señal purificada...")
    df_test = add_rolling_features(df_test, windows=[5, 15])
    df_test.fillna(0, inplace=True)
    
    # Aislar el momento actual (último ciclo visible de cada motor)
    latest_cycles = df_test.groupby('unit_id').last().reset_index()
    
    # 3. Cargar la Verdad Absoluta (RUL_FD002.txt)
    print("3. Cargando el Oráculo de la Tierra (RUL_FD002.txt)...")
    rul_path = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'RUL_FD002.txt'
    # pd.read_csv para cargar las vidas útiles restantes dictadas por la NASA
    true_rul = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['real_rul'])
    
    # Si la verdadera vida restante es <= 30, significa que ESE MOTOR iba a explotar dentro de 30 ciclos
    true_rul['failure_within_30_cycles'] = (true_rul['real_rul'] <= 30).astype(int)
    y_test_real = true_rul['failure_within_30_cycles']
    
    # 4. Carga del Modelo Tuneado
    print("4. Despertando a XGBoost (Tuned) y atacando el blind test...")
    model = joblib.load('models/xgboost_fd002_tuned.joblib')
    
    # Forzar el orden de columnas que XGBoost memorizó
    expected_features = model.feature_names_in_
    X_test_live = latest_cycles[expected_features]
    
    # 5. Predicción Definitiva
    print("5. Pronosticando a vida o muerte sobre los estatus actuales...\n")
    y_pred = model.predict(X_test_live)
    
    # Comparativa
    cm = confusion_matrix(y_test_real, y_pred)
    
    print("================== MATRIZ DE CONFUSIÓN DE TEST (FD002) ==================")
    print("                  Predicho: SANO (0)   Predicho: PELIGRO (1)")
    print(f"Real: SANO (0)  |        {cm[0][0]:<4}        |         {cm[0][1]:<4}          |")
    print(f"Real: PELIGRO (1)|        {cm[1][0]:<4}        |         {cm[1][1]:<4}          |")
    
    print("\n================ REPORTE DE CLASIFICACIÓN FINAL =================")
    print(classification_report(y_test_real, y_pred, target_names=['SANO (0)', 'PELIGRO (1)']))
    
    print("\n¡Prueba Multi-régimen superada. Motor de IA productivo analizado!")

if __name__ == "__main__":
    main()
