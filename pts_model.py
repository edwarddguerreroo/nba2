import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_absolute_error
import json
from tqdm import tqdm

class PTSMLP(nn.Module):
    def __init__(self, input_size, vocab_sizes, embedding_dim=32):
        super().__init__()
        
        # Embeddings para características categóricas
        self.pos_embedding = nn.Embedding(vocab_sizes['Pos'], embedding_dim)
        self.team_embedding = nn.Embedding(vocab_sizes['Team'], embedding_dim)
        self.opp_embedding = nn.Embedding(vocab_sizes['Opp'], embedding_dim)
        self.pos_opp_embedding = nn.Embedding(vocab_sizes['Pos_Opp'], embedding_dim)
        
        # Normalización para características numéricas
        self.input_norm = nn.LayerNorm(normalized_shape=[input_size])
        
        # Calcular tamaño total después de concatenar embeddings
        total_embedding_dim = embedding_dim * 4
        total_input_size = input_size + total_embedding_dim
        
        # Normalización después de concatenar embeddings
        self.combined_norm = nn.LayerNorm(total_input_size)
        
        # Capas fully connected con batch normalization, layer normalization y dropout
        self.fc1 = nn.Linear(total_input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.ln1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.ln2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.ln3 = nn.LayerNorm(64)
        self.dropout3 = nn.Dropout(0.2)
        
        self.output = nn.Linear(64, 1)
        
        # Guardar tamaños de vocabulario para validación
        self.vocab_sizes = vocab_sizes
        
        # Inicialización de pesos con regularización
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Inicialización He/Kaiming con un factor de escala más pequeño
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                m.weight.data *= 0.8  # Reducir la escala para mayor regularización
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)  # Reducir la varianza inicial
                    
    def forward(self, x, categorical_indices):
        # Verificar y manejar NaN en la entrada
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
            
        # Normalizar características numéricas
        batch_size = x.size(0)
        original_shape = x.shape
        
        if len(original_shape) == 3:
            x = x.reshape(-1, original_shape[-1])
            x = self.input_norm(x)
            x = x.reshape(original_shape)
        else:
            x = self.input_norm(x)
        
        # Verificar y ajustar índices categóricos
        max_values = torch.tensor([
            self.vocab_sizes['Pos'] - 1,
            self.vocab_sizes['Team'] - 1,
            self.vocab_sizes['Opp'] - 1,
            self.vocab_sizes['Pos_Opp'] - 1
        ], device=categorical_indices.device)
        
        clamped_indices = torch.zeros_like(categorical_indices)
        for i in range(4):
            clamped_indices[:, i] = torch.clamp(categorical_indices[:, i], min=0, max=max_values[i])
        
        categorical_indices = clamped_indices
        
        try:
            # Extraer y normalizar embeddings
            pos_emb = self.dropout1(self.pos_embedding(categorical_indices[:, 0]))
            team_emb = self.dropout1(self.team_embedding(categorical_indices[:, 1]))
            opp_emb = self.dropout1(self.opp_embedding(categorical_indices[:, 2]))
            pos_opp_emb = self.dropout1(self.pos_opp_embedding(categorical_indices[:, 3]))
            
            # Concatenar embeddings
            embeddings = torch.cat([pos_emb, team_emb, opp_emb, pos_opp_emb], dim=1)
            
            if len(original_shape) == 3:
                embeddings = embeddings.unsqueeze(1).expand(-1, original_shape[1], -1)
                x = torch.cat([x, embeddings], dim=2)
                x = x.reshape(batch_size, -1)
            else:
                x = torch.cat([x, embeddings], dim=1)
            
            # Normalización adicional después de concatenar
            x = self.combined_norm(x)
                
        except Exception as e:
            print(f"Error en embeddings: {e}")
            return None
            
        # Forward pass con skip connections y doble normalización
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = self.ln1(x1)
        x1 = torch.relu(x1)
        x1 = self.dropout1(x1)
        
        # Skip connection 1
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = self.ln2(x2)
        x2 = torch.relu(x2)
        x2 = self.dropout2(x2)
        
        # Skip connection 2
        x3 = self.fc3(x2)
        x3 = self.bn3(x3)
        x3 = self.ln3(x3)
        x3 = torch.relu(x3)
        x3 = self.dropout3(x3)
        
        # Capa de salida con activación suave
        out = self.output(x3)
        out = torch.relu(out)
        
        if torch.isnan(out).any():
            print("¡Advertencia! NaN detectado en la salida del modelo")
            return None
            
        return out

class PTSTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, output_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = output_dir
        
        # Configurar dispositivo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Configurar optimizador con weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler más robusto
        self.onecycle_scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.01,
            epochs=100,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        # Scheduler adicional para reducir el learning rate cuando el modelo se estanca
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Pérdida combinada: MAE + MSE
        self.criterion = lambda output, target: (
            0.7 * nn.L1Loss()(output, target) +
            0.3 * nn.MSELoss()(output, target)
        )
        
        # Early stopping con más paciencia
        self.best_val_loss = float('inf')
        self.patience = 15
        self.patience_counter = 0
        
        # Métricas
        self.train_losses = []
        self.val_losses = []
        self.train_r2 = []
        self.val_r2 = []
        self.lrs = []
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        valid_batches = 0
        
        for batch_idx, ((data, cat_indices), target) in enumerate(self.train_loader):
            try:
                # Mover datos al dispositivo
                data = data.to(self.device)
                cat_indices = cat_indices.to(self.device)
                target = target.to(self.device)
                
                # Limpiar gradientes
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data, cat_indices)
                
                # Verificar y limpiar valores no válidos
                if torch.isnan(target).any():
                    target = torch.nan_to_num(target, nan=0.0)
                
                # Calcular pérdida
                loss = self.criterion(output, target)
                
                # Backward pass con gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Actualizar pesos
                self.optimizer.step()
                
                # Actualizar learning rate con OneCycleLR
                self.onecycle_scheduler.step()
                
                total_loss += loss.item()
                valid_batches += 1
                
                # Procesar predicciones
                preds = output.detach().cpu().numpy()
                targets = target.cpu().numpy()
                
                all_preds.extend(preds)
                all_targets.extend(targets)
                
            except Exception as e:
                print(f"Error en batch {batch_idx}: {str(e)}")
                continue
        
        if valid_batches == 0:
            return float('inf'), 0.0
        
        avg_loss = total_loss / valid_batches
        r2 = r2_score(all_targets, all_preds)
        
        return avg_loss, r2
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        valid_batches = 0
        
        with torch.no_grad():
            for batch_idx, ((data, cat_indices), target) in enumerate(loader):
                try:
                    # Mover datos al dispositivo
                    data = data.to(self.device)
                    cat_indices = cat_indices.to(self.device)
                    target = target.to(self.device)
                    
                    # Verificar índices categóricos
                    if torch.any(cat_indices < 0):
                        print(f"¡Advertencia! Índices negativos detectados en batch {batch_idx}")
                        continue
                    
                    # Forward pass
                    output = self.model(data, cat_indices)
                    
                    # Verificar si el modelo devolvió None
                    if output is None:
                        print(f"¡Advertencia! Salida None en batch {batch_idx}")
                        continue
                    
                    # Verificar y manejar valores NaN en target
                    if torch.isnan(target).any():
                        target = torch.nan_to_num(target, nan=0.0)
                    
                    loss = self.criterion(output, target)
                    
                    # Verificar pérdida válida
                    if not torch.isfinite(loss):
                        print(f"¡Advertencia! Pérdida no finita en batch {batch_idx}")
                        continue
                    
                    total_loss += loss.item()
                    valid_batches += 1
                    
                    # Procesar predicciones
                    preds = output.cpu().numpy()
                    targets = target.cpu().numpy()
                    
                    # Filtrar valores no válidos
                    mask = np.isfinite(preds.flatten()) & np.isfinite(targets.flatten())
                    preds = preds.flatten()[mask]
                    targets = targets.flatten()[mask]
                    
                    if len(preds) > 0:
                        all_preds.extend(preds)
                        all_targets.extend(targets)
                        
                except Exception as e:
                    print(f"Error en batch {batch_idx}: {str(e)}")
                    continue
        
        if len(all_preds) == 0 or valid_batches == 0:
            print("¡Advertencia! No hay predicciones válidas en esta validación")
            return float('inf'), 0.0, [], []
        
        avg_loss = total_loss / valid_batches
        r2 = r2_score(all_targets, all_preds) if len(all_preds) > 1 else 0.0
        
        return avg_loss, r2, all_preds, all_targets
    
    def train(self, num_epochs=100):
        best_model_state = None
        best_r2 = -float('inf')
        
        for epoch in range(num_epochs):
            # Entrenamiento
            train_loss, train_r2 = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_r2.append(train_r2)
            
            # Validación
            val_loss, val_r2, _, _ = self.validate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_r2.append(val_r2)
            
            # Guardar learning rate actual
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lrs.append(current_lr)
            
            # Actualizar learning rate con ReduceLROnPlateau
            self.plateau_scheduler.step(val_loss)
            
            # Early stopping basado en R² en lugar de pérdida
            if val_r2 > best_r2:
                best_r2 = val_r2
                self.patience_counter = 0
                best_model_state = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.onecycle_scheduler.state_dict()
                }
                self.save_model(best_model_state)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f'Early stopping en epoch {epoch}')
                    break
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}:')
                print(f'Train Loss: {train_loss:.4f}, Train R²: {train_r2:.4f}')
                print(f'Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}')
                print(f'Learning rate: {current_lr:.6f}')
        
        # Cargar el mejor modelo antes de la evaluación final
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state['model_state_dict'])
        
        # Evaluación final
        self.evaluate()
    
    def evaluate(self):
        test_loss, test_r2, predictions, targets = self.validate(self.test_loader)
        
        # Calcular MAE
        mae = mean_absolute_error(targets, predictions)
        
        # Guardar resultados
        results = {
            'test_loss': test_loss,
            'test_r2': test_r2,
            'test_mae': mae,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_r2': self.train_r2,
            'val_r2': self.val_r2
        }
        
        # Guardar informe
        self.save_report(results)
        
        # Generar visualización
        self.plot_results(predictions, targets)
        
        return results
    
    def save_model(self, state_dict=None):
        path = os.path.join(self.output_dir, 'modelos', 'pts_mlp.pth')
        if state_dict is None:
            state_dict = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.onecycle_scheduler.state_dict()
            }
        torch.save(state_dict, path)
    
    def save_report(self, results):
        # Convertir valores NumPy a tipos nativos de Python
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        report = {
            'Métricas Finales': {
                'MAE': convert_to_serializable(results['test_mae']),
                'R²': convert_to_serializable(results['test_r2']),
                'Loss': convert_to_serializable(results['test_loss'])
            },
            'Mejor Epoch': convert_to_serializable(np.argmin(results['val_losses']) + 1),
            'Sugerencias de Mejora': self.generate_improvement_suggestions(results)
        }
        
        path = os.path.join(self.output_dir, 'informes', 'pts_training_report.txt')
        with open(path, 'w') as f:
            json.dump(report, f, indent=4)
    
    def generate_improvement_suggestions(self, results):
        suggestions = []
        
        if results['test_r2'] < 0.97:
            if results['test_mae'] > 2.0:
                suggestions.append("Considerar aumentar la complejidad del modelo (más capas o neuronas)")
            if max(results['train_losses']) - min(results['train_losses']) > 1.0:
                suggestions.append("Implementar learning rate scheduling para estabilizar el entrenamiento")
            if results['test_r2'] < 0.90:
                suggestions.append("Explorar características adicionales o técnicas de feature engineering más avanzadas")
        
        return suggestions
    
    def plot_results(self, predictions, targets):
        plt.figure(figsize=(10, 6))
        plt.scatter(targets, predictions, alpha=0.5)
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
        plt.xlabel('PTS Reales')
        plt.ylabel('PTS Predichos')
        plt.title('Predicción de Puntos (PTS)')
        
        path = os.path.join(self.output_dir, 'visualizaciones', 'pts_predictions.png')
        plt.savefig(path)
        plt.close() 