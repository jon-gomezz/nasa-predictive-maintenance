# Usa la imagen oficial y ligera de Python 3.11
FROM python:3.11-slim

# Evitar que los prints de Python se queden atascados en sub-buffers
ENV PYTHONUNBUFFERED=1

# Instalar liberías a nivel de OS (opcional pero seguro)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Establecer la carpeta de trabajo
WORKDIR /app

# Copiar el core de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código de nuestra app a la imagen
COPY . .

# Exponer el puerto de Backend (FastAPI = 8000) y de Web (Streamlit = 8501)
EXPOSE 8000 8501

# Ejecutar el orquestador principal
CMD ["python", "run_app.py"]
