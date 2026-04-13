import pandas as pd

def add_rul_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la Vida Útil Restante (RUL) exacta para cada ciclo de cada motor.
    La RUL es regresiva: el último ciclo (máximo) tiene RUL = 0.
    """
    # 1. Agrupamos por motor (unit_id) para encontrar su ciclo máximo (el momento de fallo)
    max_cycles = df.groupby('unit_id')['time_cycle'].max().reset_index()
    max_cycles.rename(columns={'time_cycle': 'max_cycle'}, inplace=True)
    
    # 2. Hacemos un merge para tener el 'max_cycle' disponible en cada fila
    df = df.merge(max_cycles, on='unit_id', how='left')
    
    # 3. Calculamos la RUL restando el ciclo máximo menos el ciclo actual del motor
    df['rul'] = df['max_cycle'] - df['time_cycle']
    
    # Limpiamos la columna auxiliar
    df.drop(columns=['max_cycle'], inplace=True)
    
    return df

def add_binary_target(df: pd.DataFrame, window_size: int = 30) -> pd.DataFrame:
    """
    Crea la columna objetivo binaria: 1 si el motor fallará en los próximos 'window_size' ciclos, 0 si no.
    """
    if 'rul' not in df.columns:
        df = add_rul_columns(df)
        
    # Asignamos 1 si la Vida Útil Restante es menor o igual a la ventana de riesgo (30)
    df['failure_within_30_cycles'] = (df['rul'] <= window_size).astype(int)
    return df

def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina las columnas que tienen varianza cero (ruido constante).
    Se usa dinámicamente: calcula la desviación estándar de la tabla de entrada.
    """
    # Filtramos para no romper columnas booleanas o IDs que por casualidad sean constantes 
    # (aunque unit_id y time_cycle ya sabemos que sí varían).
    std_dev = df.std(numeric_only=True)
    zero_variance_cols = std_dev[std_dev < 1e-4].index.tolist()
    
    # Retornamos el dataframe sin las columnas inservibles
    return df.drop(columns=zero_variance_cols)

def add_rolling_features(df: pd.DataFrame, windows: list = [5, 15]) -> pd.DataFrame:
    """
    Calcula la media móvil y la desviación estándar para las ventanas de tiempo especificadas.
    Es vital hacer el groupby por 'unit_id' para no mezclar la ventana de un motor con el final del anterior.
    """
    cols_to_roll = [c for c in df.columns if c.startswith('sensor_') or c.startswith('op_setting_')]
    
    # Nos aseguramos de mantener un índice natural en caso de desorden
    df = df.sort_values(['unit_id', 'time_cycle']).copy()
    
    for w in windows:
        # 1. Rolling Mean (Media móvil)
        # min_periods=1 ayuda a no tener tantos NaNs en los primeros ciclos del motor
        rolling_mean = df.groupby('unit_id')[cols_to_roll].rolling(window=w, min_periods=1).mean()
        # Reset index alignment
        rolling_mean = rolling_mean.reset_index(level=0, drop=True)
        # Rename columns to avoid collision
        rolling_mean.columns = [f"{c}_mean_{w}" for c in cols_to_roll]
        
        # 2. Rolling Standard Deviation (Varianza local)
        # min_periods=2 porque la std necesita al menos 2 datos para calcularse
        rolling_std = df.groupby('unit_id')[cols_to_roll].rolling(window=w, min_periods=2).std()
        rolling_std = rolling_std.reset_index(level=0, drop=True)
        rolling_std.columns = [f"{c}_std_{w}" for c in cols_to_roll]
        
        # Juntamos estas nuevas features al dataframe principal
        df = pd.concat([df, rolling_mean, rolling_std], axis=1)
        
    return df

class ConditionNormalizer:
    """
    Clase para aislar y normalizar los regímenes de vuelo (Clustering K-Means).
    Es vital hacer fit en Train y solo transform en Test.
    """
    def __init__(self, n_clusters=6):
        self.n_clusters = n_clusters
        from sklearn.cluster import KMeans
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scalers = {} 
        self.sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
        self.settings_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Aprender los clusters basándose puramente en altura/Mach/Throttle
        df['condition_cluster'] = self.kmeans.fit_predict(df[self.settings_cols])
        
        # 2. Normalizar cada sensor AISLADAMENTE dentro de su propio cluster
        from sklearn.preprocessing import StandardScaler
        for cluster_id in range(self.n_clusters):
            scaler = StandardScaler()
            mask = df['condition_cluster'] == cluster_id
            
            if mask.sum() > 0:
                # Extraemos y normalizamos
                cols_to_scale = [c for c in self.sensor_cols if c in df.columns]
                df.loc[mask, cols_to_scale] = scaler.fit_transform(df.loc[mask, cols_to_scale])
                
                # Guardamos el scaler exacto que aprendió la temperatura/presión "normal" en este clima
                self.scalers[cluster_id] = scaler
                
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Asignar el vuelo ciego a uno de los 6 regímenes conocidos
        df['condition_cluster'] = self.kmeans.predict(df[self.settings_cols])
        
        # 2. Aplicar el escalado específico aprendido en Train
        for cluster_id in range(self.n_clusters):
            mask = df['condition_cluster'] == cluster_id
            cols_to_scale = [c for c in self.sensor_cols if c in df.columns]
            
            if mask.sum() > 0 and cluster_id in self.scalers:
                scaler = self.scalers[cluster_id]
                df.loc[mask, cols_to_scale] = scaler.transform(df.loc[mask, cols_to_scale])
                
        return df

