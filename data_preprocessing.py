import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import json

class NBADataset(Dataset):
    def __init__(self, features, targets, categorical_indices):
        # Asegurar que los features sean 2D
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
            
        # Convertir a tensores y asegurar dimensiones correctas
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets).reshape(-1, 1)
        self.categorical_indices = torch.LongTensor(categorical_indices)
        
        # Verificar y ajustar dimensiones
        if len(self.features.shape) != 2:
            raise ValueError(f"Features deben ser 2D, pero tienen forma {self.features.shape}")
        if len(self.targets.shape) != 2:
            raise ValueError(f"Targets deben ser 2D, pero tienen forma {self.targets.shape}")
        if len(self.categorical_indices.shape) != 2:
            raise ValueError(f"Categorical indices deben ser 2D, pero tienen forma {self.categorical_indices.shape}")
            
        # Asegurar que las dimensiones sean consistentes
        if self.features.shape[0] != self.targets.shape[0] or self.features.shape[0] != self.categorical_indices.shape[0]:
            raise ValueError("El número de muestras debe ser igual para features, targets y categorical_indices")
            
        # Verificar que los índices categóricos sean válidos
        if torch.any(self.categorical_indices < 0):
            raise ValueError("Los índices categóricos no pueden ser negativos")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Asegurar que los tensores tengan la forma correcta
        features = self.features[idx]
        cat_indices = self.categorical_indices[idx]
        target = self.targets[idx]
        
        # Verificar NaN
        if torch.isnan(features).any():
            features = torch.nan_to_num(features, nan=0.0)
        if torch.isnan(target).any():
            target = torch.nan_to_num(target, nan=0.0)
            
        return (features, cat_indices), target

    def get_feature_dim(self):
        return self.features.shape[1]

    def get_num_categories(self):
        return self.categorical_indices.shape[1]

    def get_vocab_sizes(self):
        return {
            'Pos': self.categorical_indices[:, 0].max().item() + 1,
            'Team': self.categorical_indices[:, 1].max().item() + 1,
            'Opp': self.categorical_indices[:, 2].max().item() + 1,
            'Pos_Opp': self.categorical_indices[:, 3].max().item() + 1
        }

class DataPreprocessor:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        # Convertir a ruta absoluta
        self.output_dir = os.path.abspath(output_dir)
        self.scaler = StandardScaler()
        self.categorical_encoders = {}
        
        # Crear directorios necesarios con rutas absolutas
        for subdir in ['modelos', 'visualizaciones', 'informes']:
            dir_path = os.path.join(self.output_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Creando directorio: {dir_path}")

    def load_and_clean_data(self):
        """Carga y limpia los datos iniciales."""
        # Cargar datos
        df = pd.read_csv(self.data_path)
        
        # Verificar que las columnas necesarias existan
        required_cols = ['Player', 'Date', 'Team', 'Opp', 'Pos', 'PTS', 'TRB', 'AST']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print("Columnas disponibles en el dataset:", df.columns.tolist())
            raise ValueError(f"Columnas requeridas faltantes: {missing_cols}")
        
        # Convertir fecha a datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Ordenar por jugador y fecha
        df = df.sort_values(['Player', 'Date'])
        
        # Limpiar columna Result (extraer solo W/L)
        if 'Result' in df.columns:
            df['Result'] = df['Result'].str.extract(r'([WL])')
        
        # Convertir columnas de porcentajes a números
        pct_cols = ['FG%', '2P%', '3P%', 'FT%', 'TS%']
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = df[col].replace('', '0%')
                df[col] = pd.to_numeric(df[col].str.rstrip('%'), errors='coerce') / 100
                df[col] = df[col].clip(0, 1)
        
        # Identificar columnas numéricas y categóricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = ['Pos', 'Team', 'Opp']
        
        # Manejo robusto de valores atípicos usando MAD
        def handle_outliers_mad(df, col, group_col):
            """Maneja outliers usando MAD por grupo."""
            df = df.copy()
            for group_name, group in df.groupby(group_col):
                median = group[col].median()
                mad = np.median(np.abs(group[col] - median))
                lower_bound = median - 3.5 * mad
                upper_bound = median + 3.5 * mad
                mask = group.index
                
                # Convertir los valores al tipo de dato de la columna original
                original_dtype = df[col].dtype
                clipped_values = df.loc[mask, col].clip(lower_bound, upper_bound)
                df.loc[mask, col] = clipped_values.astype(original_dtype)
            
            return df
        
        # Aplicar manejo de outliers por posición y jugador
        for col in ['PTS', 'TRB', 'AST']:
            # Primero por posición
            df = handle_outliers_mad(df, col, 'Pos')
            # Luego por jugador
            df = handle_outliers_mad(df, col, 'Player')
        
        # Estrategia de imputación mejorada para valores nulos
        for col in numeric_cols:
            # 1. Intentar imputar con la mediana de los últimos 5 juegos del jugador
            df[col] = df.groupby('Player')[col].transform(
                lambda x: x.fillna(x.rolling(window=5, min_periods=1).median())
            )
            
            # 2. Si aún hay nulos, usar la mediana de la posición del jugador
            df[col] = df.groupby('Pos')[col].transform(
                lambda x: x.fillna(x.median())
            )
            
            # 3. Si aún hay nulos, usar la mediana global
            df[col] = df[col].fillna(df[col].median())
        
        # Imputación mejorada para categóricas
        for col in categorical_cols:
            # 1. Usar la moda del jugador
            df[col] = df.groupby('Player')[col].transform(
                lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else None)
            )
            
            # 2. Si aún hay nulos, usar la moda global
            df[col] = df[col].fillna(df[col].mode().iloc[0])
        
        return df

    def create_temporal_features(self, df):
        """Crea características temporales por jugador."""
        features = []
        
        for player in df['Player'].unique():
            player_data = df[df['Player'] == player].copy()
            
            # Promedios móviles
            for window in [3, 5, 10, 15]:
                cols = ['PTS', 'TRB', 'AST', 'MP', 'FG', 'FGA', '3P', '3PA', 
                       'FT', 'FTA', 'STL', 'BLK', 'TOV', 'PF', 'GmSc', 'BPM', '+/-']
                
                for col in cols:
                    player_data[f'{col}_avg_{window}'] = player_data[col].rolling(window=window, min_periods=1).mean()
            
            # Tendencias
            for stat in ['PTS', 'TRB', 'AST']:
                player_data[f'tendencia_{stat.lower()}_5'] = (
                    player_data[f'{stat}_avg_5'] - 
                    player_data[f'{stat}_avg_5'].shift(5)
                )
            
            # Máximos recientes
            for stat in ['PTS', 'TRB', 'AST']:
                player_data[f'max_{stat.lower()}_5'] = player_data[stat].rolling(window=5, min_periods=1).max()
            
            features.append(player_data)
        
        return pd.concat(features, ignore_index=True)

    def create_efficiency_features(self, df):
        """Crea características de eficiencia y ratios."""
        # Evitar división por cero reemplazando ceros con un valor pequeño
        eps = 1e-7
        mp_safe = df['MP'].replace(0, eps)
        
        # Tiros por minuto
        df['FG_per_MP'] = (df['FG'] / mp_safe).clip(-1e6, 1e6)
        df['FGA_per_MP'] = (df['FGA'] / mp_safe).clip(-1e6, 1e6)
        df['3P_per_MP'] = (df['3P'] / mp_safe).clip(-1e6, 1e6)
        df['FT_per_MP'] = (df['FT'] / mp_safe).clip(-1e6, 1e6)
        
        # Eficiencia ajustada
        df['TS_adj'] = (df['TS%'] * df['MP']).clip(-1e6, 1e6)
        
        # Ratio de uso
        df['usage_ratio'] = ((df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / mp_safe).clip(-1e6, 1e6)
        
        # Rebotes por minuto
        df['ORB_per_MP'] = (df['ORB'] / mp_safe).clip(-1e6, 1e6)
        df['DRB_per_MP'] = (df['DRB'] / mp_safe).clip(-1e6, 1e6)
        df['TRB_per_MP'] = (df['TRB'] / mp_safe).clip(-1e6, 1e6)
        
        # Asistencias por turno perdido
        tov_safe = df['TOV'].replace(0, eps)
        df['AST_per_TOV'] = (df['AST'] / tov_safe).clip(-1e6, 1e6)
        
        return df

    def create_context_features(self, df):
        """Crea características de contexto."""
        # Dummy de titularidad reciente
        # Convertir GS a numérico: '*' -> 0, otros valores -> 1
        df['GS_numeric'] = (df['GS'] != '*').astype(int)
        
        # Calcular promedios móviles manteniendo el índice original
        df['GS_avg_5'] = df.groupby('Player', group_keys=False).apply(
            lambda x: x['GS_numeric'].rolling(window=5, min_periods=1).mean()
        )
        
        # Diferencial vs oponente
        df['plus_minus_avg_5'] = df.groupby(['Player', 'Opp'], group_keys=False).apply(
            lambda x: x['+/-'].rolling(window=5, min_periods=1).mean()
        )
        
        # Victorias recientes
        df['Result_binary'] = (df['Result'] == 'W').astype(int)
        df['win_rate_5'] = df.groupby('Player', group_keys=False).apply(
            lambda x: x['Result_binary'].rolling(window=5, min_periods=1).mean()
        )
        
        # Eliminar columna temporal
        df = df.drop('GS_numeric', axis=1)
        
        return df

    def create_advanced_features(self, df):
        """Crea características avanzadas."""
        # Evitar división por cero
        eps = 1e-7
        mp_safe = df['MP'].replace(0, eps)
        
        # Contribución defensiva
        df['defensive_contribution'] = ((df['STL'] + df['BLK']) / mp_safe).clip(-1e6, 1e6)
        
        # Impacto neto ajustado
        df['net_impact_adj'] = (df['BPM'] * df['MP'] / 48).clip(-1e6, 1e6)
        
        # Game Score por minuto
        df['GmSc_per_MP'] = (df['GmSc'] / mp_safe).clip(-1e6, 1e6)
        
        # Interacciones entre variables
        df['AST_STL_interaction'] = (df['AST'] * df['STL']).clip(-1e6, 1e6)
        df['TRB_BLK_interaction'] = (df['TRB'] * df['BLK']).clip(-1e6, 1e6)
        df['PTS_TS_interaction'] = (df['PTS'] * df['TS%']).clip(-1e6, 1e6)
        
        return df

    def create_categorical_features(self, df):
        """Crea y codifica características categóricas."""
        # Frecuencia de enfrentamiento
        df['opp_frequency'] = df.groupby(['Player', 'Opp']).cumcount() + 1
        
        # Posición-oponente
        df['Pos_Opp'] = df['Pos'] + '-' + df['Opp']
        
        # Definir tamaños máximos de vocabulario
        max_vocab_sizes = {
            'Pos': 5,  # Número típico de posiciones en baloncesto
            'Team': 30,  # Número de equipos en la NBA
            'Opp': 30,  # Número de equipos en la NBA
            'Pos_Opp': 210  # Aumentado para acomodar todas las combinaciones únicas
        }
        
        # Codificación de variables categóricas
        categorical_cols = ['Pos', 'Team', 'Opp', 'Pos_Opp']
        for col in categorical_cols:
            # Obtener valores únicos ordenados
            unique_values = sorted(df[col].unique())
            
            # Verificar si excedemos el tamaño máximo
            if len(unique_values) > max_vocab_sizes[col]:
                print(f"¡Advertencia! {col} tiene más valores únicos ({len(unique_values)}) que el máximo permitido ({max_vocab_sizes[col]})")
                
                if col == 'Pos':
                    # Para Pos, agrupar valores menos comunes en "OTHER"
                    value_counts = df[col].value_counts()
                    top_values = value_counts.head(max_vocab_sizes[col] - 1).index.tolist()
                    df[col] = df[col].apply(lambda x: x if x in top_values else "OTHER")
                    unique_values = sorted(df[col].unique())
                else:
                    # Para otras columnas, tomar los valores más frecuentes
                    value_counts = df[col].value_counts()
                    top_values = value_counts.head(max_vocab_sizes[col]).index.tolist()
                    unique_values = sorted(top_values)
            
            # Crear diccionario de mapeo empezando desde 0
            self.categorical_encoders[col] = {val: idx for idx, val in enumerate(unique_values)}
            
            # Valor por defecto para valores desconocidos
            default_idx = len(unique_values) - 1
            
            # Aplicar codificación con manejo de valores desconocidos
            df[f'{col}_encoded'] = df[col].map(lambda x: self.categorical_encoders[col].get(x, default_idx))
            
            # Verificar valores nulos
            null_count = df[f'{col}_encoded'].isnull().sum()
            if null_count > 0:
                print(f"¡Advertencia! {null_count} valores nulos encontrados en {col}_encoded")
                # Reemplazar valores nulos con el último índice
                df[f'{col}_encoded'] = df[f'{col}_encoded'].fillna(default_idx)
            
            # Asegurar que todos los índices estén dentro del rango válido
            df[f'{col}_encoded'] = df[f'{col}_encoded'].clip(0, max_vocab_sizes[col] - 1)
            
            # Guardar el número de categorías únicas
            self.categorical_encoders[f'{col}_size'] = max_vocab_sizes[col]
            
            # Verificar rango de índices final
            min_idx = df[f'{col}_encoded'].min()
            max_idx = df[f'{col}_encoded'].max()
            print(f"Rango final de índices para {col}: [{min_idx}, {max_idx}] de {max_vocab_sizes[col]} posibles")
        
        return df

    def prepare_sequences(self, df, target_col, sequence_length=5):
        """Prepara secuencias para el modelo LSTM."""
        sequences = []
        targets = []
        categorical_indices = []
        
        # Obtener solo columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols 
                       if col not in [target_col, 'Result_binary'] and 
                       not col.endswith('_encoded')]
        
        # Columnas categóricas codificadas
        cat_cols = ['Pos_encoded', 'Team_encoded', 'Opp_encoded', 'Pos_Opp_encoded']
        
        # Verificar que todas las columnas necesarias existan
        missing_cols = [col for col in feature_cols + cat_cols + [target_col] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes en el DataFrame: {missing_cols}")
        
        # Definir tamaños máximos de vocabulario
        max_vocab_sizes = {
            'Pos': 5,
            'Team': 30,
            'Opp': 30,
            'Pos_Opp': 150
        }
        
        for player in df['Player'].unique():
            player_data = df[df['Player'] == player].copy()
            
            for i in range(len(player_data) - sequence_length + 1):
                # Obtener secuencia y target
                sequence = player_data[feature_cols].iloc[i:i+sequence_length].values
                target = player_data[target_col].iloc[i+sequence_length-1]
                cat_indices = player_data[cat_cols].iloc[i+sequence_length-1].values
                
                # Verificar valores no válidos
                if np.isnan(sequence).any() or np.isnan(target) or np.isnan(cat_indices).any():
                    continue
                
                # Asegurar que los índices categóricos estén dentro de los límites
                valid_sequence = True
                for j, col in enumerate(cat_cols):
                    base_col = col.replace('_encoded', '')
                    if cat_indices[j] < 0 or cat_indices[j] >= max_vocab_sizes[base_col]:
                        valid_sequence = False
                        break
                
                if not valid_sequence:
                    continue
                
                sequences.append(sequence)
                targets.append(target)
                categorical_indices.append(cat_indices)
        
        if not sequences:
            raise ValueError("No se encontraron secuencias válidas después del filtrado")
        
        # Convertir a arrays numpy
        sequences = np.array(sequences)
        targets = np.array(targets)
        categorical_indices = np.array(categorical_indices)
        
        # Verificación final de índices categóricos
        for i, col in enumerate(cat_cols):
            base_col = col.replace('_encoded', '')
            categorical_indices[:, i] = np.clip(
                categorical_indices[:, i],
                0,
                max_vocab_sizes[base_col] - 1
            )
            
            # Imprimir estadísticas de los índices
            min_idx = categorical_indices[:, i].min()
            max_idx = categorical_indices[:, i].max()
            print(f"Rango final de índices para {col}: [{min_idx}, {max_idx}] de {max_vocab_sizes[base_col]} posibles")
        
        return sequences, targets, categorical_indices

    def split_data(self, df, target_col, sequence_length=5):
        """Divide los datos en conjuntos de entrenamiento, validación y prueba."""
        # Ordenar por fecha
        df = df.sort_values('Date')
        
        # Calcular índices de división
        train_idx = int(len(df) * 0.7)
        val_idx = int(len(df) * 0.85)
        
        # Dividir datos
        train_df = df.iloc[:train_idx]
        val_df = df.iloc[train_idx:val_idx]
        test_df = df.iloc[val_idx:]
        
        # Preparar secuencias para cada conjunto
        X_train, y_train, cat_train = self.prepare_sequences(train_df, target_col, sequence_length)
        X_val, y_val, cat_val = self.prepare_sequences(val_df, target_col, sequence_length)
        X_test, y_test, cat_test = self.prepare_sequences(test_df, target_col, sequence_length)
        
        # Normalizar características
        X_train = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        return (X_train, y_train, cat_train), (X_val, y_val, cat_val), (X_test, y_test, cat_test)

    def create_data_loaders(self, train_data, val_data, test_data, batch_size=32):
        """Crea DataLoaders para los conjuntos de datos."""
        train_dataset = NBADataset(train_data[0], train_data[1], train_data[2])
        val_dataset = NBADataset(val_data[0], val_data[1], val_data[2])
        test_dataset = NBADataset(test_data[0], test_data[1], test_data[2])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader

    def save_preprocessing_info(self):
        """Guarda información del preprocesamiento."""
        info = {
            'scaler_params': {
                'mean_': self.scaler.mean_.tolist(),
                'scale_': self.scaler.scale_.tolist()
            },
            'categorical_encoders': self.categorical_encoders
        }
        
        with open(os.path.join(self.output_dir, 'modelos', 'preprocessing_info.json'), 'w') as f:
            json.dump(info, f)

    def process_data(self, target_col):
        """Procesa los datos completos."""
        # Cargar y limpiar datos
        df = self.load_and_clean_data()
        
        # Verificar que la columna objetivo exista
        if target_col not in df.columns:
            raise ValueError(f"La columna objetivo '{target_col}' no existe en el dataset")
        
        # Crear características
        df = self.create_temporal_features(df)
        df = self.create_efficiency_features(df)
        df = self.create_context_features(df)
        df = self.create_advanced_features(df)
        df = self.create_categorical_features(df)
        
        # Dividir datos
        train_data, val_data, test_data = self.split_data(df, target_col)
        
        # Crear data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders(
            train_data, val_data, test_data
        )
        
        # Guardar información de preprocesamiento
        self.save_preprocessing_info()
        
        return train_loader, val_loader, test_loader 