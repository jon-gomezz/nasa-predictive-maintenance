import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Agregar src path para import correcto
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.api import app

# Inicializar TestClient de FastAPI
client = TestClient(app)

def test_health_check():
    """Testea la salud basica de la API y que los modelos hayan cargado en RAM"""
    with TestClient(app) as local_client:
        response = local_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        # Aseguramos que los modelos pesados se montaron bien en el startup event
        assert data["model_loaded"] is True

def test_predict_endpoint_missing_file():
    """Testea el manejo de errores si se manda una petición POST sin el csv"""
    response = client.post("/predict")
    # Deberia devolver un Error 422 Unprocessable Entity que levanta el Pydantic/FastAPI
    assert response.status_code == 422

# ---- OPCIONAL SI HUBIESE UN ARCHIVO RAW -----
# def test_predict_endpoint_success():
#    with open("data/raw/test_FD002.txt", "rb") as f:
#        response = client.post("/predict", files={"file": ("test_FD002.txt", f, "text/plain")})
#    assert response.status_code == 200
#    data = response.json()
#    assert "analisis" in data
#    assert len(data["analisis"]) > 0
#    assert "unit_id" in data["analisis"][0]
