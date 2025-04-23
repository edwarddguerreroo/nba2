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

class TRBLSTM(nn.Module):
    def __init__(self, input_size, vocab_sizes=None, hidden_size=100, num_layers=2, embedding_dim=8):
        super(TRBLSTM, self).__init__()
        
        if vocab_sizes is None:
            vocab_sizes = {'Pos': 5, 'Team': 30, 'Opp': 30, 'Pos_Opp': 150}
        
        # Embeddings para características categóricas
        self.pos_embedding = nn.Embedding(vocab_sizes['Pos'], embedding_dim)
        self.team_embedding = nn.Embedding(vocab_sizes['Team'], embedding_dim)
        self.opp_embedding = nn.Embedding(vocab_sizes['Opp'], embedding_dim)
        self.pos_opp_embedding = nn.Embedding(vocab_sizes['Pos_Opp'], embedding_dim)
        
        # Calcular tamaño de entrada total
        total_embedding_size = embedding_dim * 4
        total_input_size = input_size + total_embedding_size
        
        # Normalización de entrada
        self.input_norm = nn.LayerNorm(total_input_size)
        
        # LSTM bidireccional con dropout
        self.lstm = nn.LSTM(
            input_size=total_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Atención para secuencias
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Capas de salida con regularización
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def attention_net(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size * 2)
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights
        
    def forward(self, x, categorical_indices):
        # Extraer embeddings con dropout
        pos_emb = nn.functional.dropout(self.pos_embedding(categorical_indices[:, 0]), p=0.1, training=self.training)
        team_emb = nn.functional.dropout(self.team_embedding(categorical_indices[:, 1]), p=0.1, training=self.training)
        opp_emb = nn.functional.dropout(self.opp_embedding(categorical_indices[:, 2]), p=0.1, training=self.training)
        pos_opp_emb = nn.functional.dropout(self.pos_opp_embedding(categorical_indices[:, 3]), p=0.1, training=self.training)
        
        # Concatenar embeddings
        embeddings = torch.cat([pos_emb, team_emb, opp_emb, pos_opp_emb], dim=1)
        
        # Asegurar que x tenga la forma correcta (batch_size, seq_len, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Expandir embeddings para que coincida con la longitud de la secuencia
        embeddings = embeddings.unsqueeze(1).expand(-1, x.size(1), -1)
        
        # Concatenar con características numéricas y normalizar
        x = torch.cat([x, embeddings], dim=2)
        x = self.input_norm(x)
        
        # Pasar por LSTM bidireccional
        lstm_out, _ = self.lstm(x)
        
        # Aplicar atención
        context, _ = self.attention_net(lstm_out)
        
        # Predicción final con activación suave
        out = self.fc(context)
        out = torch.relu(out)
        
        return out

class TRBTrainer:
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
                    'scheduler_state_dict': self.onecycle_scheduler.state_dict(),
                    'epoch': epoch,
                    'val_r2': val_r2
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
    
    def save_model(self, model_state):
        path = os.path.join(self.output_dir, 'modelos', 'trb_lstm.pth')
        torch.save(model_state, path)
    
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
        
        path = os.path.join(self.output_dir, 'informes', 'trb_training_report.txt')
        with open(path, 'w') as f:
            json.dump(report, f, indent=4)
    
    def generate_improvement_suggestions(self, results):
        suggestions = []
        
        if results['test_r2'] < 0.97:
            if results['test_mae'] > 1.5:
                suggestions.append("Considerar aumentar el número de capas LSTM o el tamaño del hidden state")
            if max(results['train_losses']) - min(results['train_losses']) > 1.0:
                suggestions.append("Implementar learning rate scheduling para estabilizar el entrenamiento")
            if results['test_r2'] < 0.90:
                suggestions.append("Explorar secuencias más largas o características adicionales")
        
        return suggestions
    
    def plot_results(self, predictions, targets):
        plt.figure(figsize=(10, 6))
        plt.scatter(targets, predictions, alpha=0.5)
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
        plt.xlabel('TRB Reales')
        plt.ylabel('TRB Predichos')
        plt.title('Predicción de Rebotes (TRB)')
        
        path = os.path.join(self.output_dir, 'visualizaciones', 'trb_predictions.png')
        plt.savefig(path)
        plt.close() 