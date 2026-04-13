import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.config import RAW_DATA_DIR, ALL_COLUMNS

def load_cmapss_data(filename: str) -> pd.DataFrame:
    """
    Carga un archivo crudo de C-MAPSS (train o test) y le asigna 
    los nombres de columnas correctos.
    Retorna un DataFrame de Pandas.
    """
    filepath = RAW_DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
        
    df = pd.read_csv(
        filepath, 
        sep=r"\s+", 
        header=None, 
        names=ALL_COLUMNS
    )
    return df

def load_rul_data(filename: str) -> pd.DataFrame:
    """
    Carga el archivo RUL original (vida útil restante real para el test set).
    Agrega la columna 'unit_id' implícita por el orden de los datos.
    """
    filepath = RAW_DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
        
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=["rul_real"])
    
    # En los archivos RUL (test), la fila i corresponde a la unit_id = i + 1
    df['unit_id'] = range(1, len(df) + 1)
    
    # Reordenamos columnas para tener unit_id primero por convención
    return df[['unit_id', 'rul_real']]

def get_train_val_split(df: pd.DataFrame, val_size: float = 0.2, random_state: int = 42):
    """
    Realiza una separación Train/Validation asegurando que los registros 
    de una misma unidad de motor (unit_id) no se filtren (leakage) 
    entre ambos conjuntos. Usamos GroupShuffleSplit para preservar motores completos.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    
    # Generamos los índices para el split basado en la columna unit_id
    train_idx, val_idx = next(gss.split(df, groups=df['unit_id']))
    
    train_split = df.iloc[train_idx].copy()
    val_split = df.iloc[val_idx].copy()
    
    return train_split, val_split
