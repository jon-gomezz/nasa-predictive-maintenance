import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data import load_cmapss_data
from src.features import add_rolling_features

def main():
    print("====== EXAMEN FINAL: SET DE TEST CIEGO (SIMULACIÓN DE PRODUCCIÓN) ======\n")
    
    # 1. Cargar el último punto de los motores "en vivo" (test_FD001.txt)
    print("1. Cargando telemetría de turbinas activas...")
    df_test = load_cmapss_data('test_FD001.txt')
    
    # Añadimos memoria temporal antes de filtrar, 
    # dejando que el motor acumule "el histórico del vuelo" en los rolling
    print("   -> Calculando ventanas deslizantes...")
    df_test = add_rolling_features(df_test, windows=[5, 15])
    df_test.fillna(0, inplace=True)
    
    # En la vida real, solo puedes predecir con los datos del "ahora mismo" (el último ciclo)
    # Agrupamos por unit_id y cogemos la última fila que tengamos de cada uno
    latest_cycles = df_test.groupby('unit_id').last().reset_index()
    
    # 2. Cargar la verdad absoluta para poder medirnos (RUL real dictada por la NASA)
    print("2. Cargando la Realidad a futuro (RUL_FD001.txt)...")
    rul_path = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'RUL_FD001.txt'
    # El fichero RUL contiene 1 columna con la vida útil restante para el motor 1, 2, 3...
    true_rul = pd.read_csv(rul_path, delim_whitespace=True, header=None, names=['real_rul'])
    
    # Si la verdadera vida restante es <= 30, significa que ESE MOTOR IBA A EXPLOTAR dentro de 30 ciclos
    true_rul['failure_within_30_cycles'] = (true_rul['real_rul'] <= 30).astype(int)
    
    y_test_real = true_rul['failure_within_30_cycles']
    
    # 3. Carga del Modelo
    print("3. Despertando al modelo de IA y filtrando sensores...")
    model = joblib.load('models/xgboost_baseline.joblib')
    
    # ¡Cuidado con el Data Leakage de características!
    # El modelo fue entrenado con ciertas columnas (porque nosotros borramos las constantes en Train)
    # Simplemente forzamos al test set a usar estrictamente esas mismas columnas que espera XGBoost
    expected_features = model.feature_names_in_
    X_test_live = latest_cycles[expected_features]
    
    # 4. Predicción a vida o muerte
    print("4. Pronosticando a vida o muerte sobre los estatus actuales...\n")
    y_pred = model.predict(X_test_live)
    
    # Comparativa
    cm = confusion_matrix(y_test_real, y_pred)
    
    print("================== MATRIZ DE CONFUSIÓN DE TEST ==================")
    print("                  Predicho: SANO (0)   Predicho: PELIGRO (1)")
    print(f"Real: SANO (0)  |        {cm[0][0]:<4}        |         {cm[0][1]:<4}          |")
    print(f"Real: PELIGRO (1)|        {cm[1][0]:<4}        |         {cm[1][1]:<4}          |")
    
    print("\n================ REPORTE DE CLASIFICACIÓN FINAL =================")
    print(classification_report(y_test_real, y_pred, target_names=['SANO (0)', 'PELIGRO (1)']))
    
    print("\n¡Despliegue simulado y validado!")

if __name__ == "__main__":
    main()
