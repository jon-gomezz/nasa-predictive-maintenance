import subprocess
import sys
import os

def run_step(comando, descripcion):
    print(f"\n=======================================================")
    print(f"🚀 INICIANDO: {descripcion}")
    print(f"=======================================================")
    resultado = subprocess.run(comando, shell=True)
    if resultado.returncode != 0:
        print(f"\n❌ Falla Crítica en el pipeline deteniendo la ejecución: {descripcion}")
        sys.exit(1)
    print(f"✅ Completado con éxito: {descripcion}\n")

def main():
    print("-------------------------------------------------------")
    print("      AUTOMATED MLOPS PIPELINE: PREDICTIVE MAINTENANCE  ")
    print("-------------------------------------------------------\n")
    
    # Asegurar que estamos en el root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Tests Unitarios (CI/CD)
    run_step("pytest tests/ -v", "Pruebas Unitarias del Microservicio y Modelos (Pytest)")
    
    # 2. Entrenamientos Reproducibles
    run_step(f"{sys.executable} src/train_fd002.py", "Training Pipeline FD002 (K-Means + Optuna + XGBoost)")
    
    # 3. Examen Final Ciego
    run_step(f"{sys.executable} src/evaluate_test_fd002.py", "Test Ciego Oculto FD002")
    
    print("=======================================================")
    print("🏆 PIPELINE E2E SUPERADO CON ÉXITO. EL SISTEMA ESTÁ LISTO PARA PRODUCCIÓN.")
    print("Para levantar el backend y visualizador ejecuta: python run_app.py")
    print("=======================================================")

if __name__ == "__main__":
    main()
