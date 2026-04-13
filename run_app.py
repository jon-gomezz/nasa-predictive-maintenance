import subprocess
import time
import sys
import os

def main():
    print("Iniciando secuencia de ignición del Servidor Turbofán MLOps...")
    
    # Prender el servidor backend (FastAPI) en el puerto 8000
    fastapi_process = subprocess.Popen(
        ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"],
        env=os.environ.copy()
    )
    
    # 2 segundos para asegurar al backend
    time.sleep(2)
    
    # Prender el visualizador (Streamlit) en el puerto 8501
    streamlit_process = subprocess.Popen(
        ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"],
        env=os.environ.copy()
    )

    try:
        # Mantener vivos a ambos hasta que el Usuario de CTRL+C
        fastapi_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\nApagando clústeres de servidores...")
        fastapi_process.terminate()
        streamlit_process.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main()
