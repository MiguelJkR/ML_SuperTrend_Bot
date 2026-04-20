# ML SuperTrend v51 - Google Colab Training Notebook
# Copia cada seccion como una celda separada en Colab
# Runtime \u2192 Change runtime type \u2192 GPU (T4)


# ============================================================
# CELDA 1: Setup
# ============================================================
# ============ CELDA 1: Instalar dependencias ============
!pip install torch torchvision torchaudio
!pip install yfinance pandas numpy scikit-learn

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import os

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")


# ============================================================
# CELDA 2: Modelo
# ============================================================
# ============ CELDA 2: Definir arquitectura del modelo ============
# (Misma arquitectura que lstm_predictor.py pero standalone)

class MultiHeadTemporalAttention(nn.Module):
    """Multi-Head Temporal Self-Attention (4 cabezas)."""
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H):
        batch_size, seq_len, _ = H.shape
        query_input = H[:, -1:, :]
        Q = self.W_q(query_input).view(batch_size, 1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(H).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(H).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, V).squeeze(2)
        context = context.reshape(batch_size, self.hidden_size)
        context = self.W_o(context)
        avg_weights = weights.squeeze(2).mean(dim=1)
        return context, avg_weights


class TorchLSTMGPU(nn.Module):
    """LSTM + Multi-Head Attention para Colab."""
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.attention = MultiHeadTemporalAttention(hidden_size, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        context = self.layer_norm(context)
        return self.classifier(context).squeeze(-1)

print("Modelo definido: TorchLSTMGPU (128h, 4-head MHA, 2 layers)")
params = sum(p.numel() for p in TorchLSTMGPU().parameters())
print(f"Parametros totales: {params:,}")


# ============================================================
# CELDA 3: Datos
# ============================================================
# ============ CELDA 3: Descargar datos de entrenamiento ============
# Usa yfinance para descargar datos historicos GRATUITOS

PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "XAUUSD": "GC=F",
}

PERIOD = "2y"        # 2 anios de datos
INTERVAL = "1h"      # Velas de 1 hora
SEQ_LENGTH = 30      # Secuencia de 30 bars
PRED_HORIZON = 5     # Predecir 5 bars adelante

all_sequences = []
all_labels = []

for name, ticker in PAIRS.items():
    print(f"Descargando {name} ({ticker})...")
    try:
        data = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False)
        if data is None or len(data) < 100:
            print(f"  [SKIP] Pocos datos para {name}")
            continue

        # Calcular features basicas
        close = data['Close'].values.flatten()
        high = data['High'].values.flatten()
        low = data['Low'].values.flatten()
        volume = data['Volume'].values.flatten() if 'Volume' in data.columns else np.ones(len(close))

        # RSI
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().values
        avg_loss = pd.Series(loss).rolling(14).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - 100 / (1 + rs)

        # ATR
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
        atr = pd.Series(tr).rolling(14).mean().values

        # MACD
        ema12 = pd.Series(close).ewm(span=12).mean().values
        ema26 = pd.Series(close).ewm(span=26).mean().values
        macd = ema12 - ema26

        # ADX (simplificado)
        adx = pd.Series(np.abs(delta)).rolling(14).mean().values / (atr + 1e-10) * 100

        # Normalizar
        close_mean = pd.Series(close).rolling(50).mean().values
        close_std = pd.Series(close).rolling(50).std().values
        close_std[close_std < 1e-6] = 1e-6

        features = np.column_stack([
            (close - close_mean) / close_std,  # close_norm
            rsi / 100.0,                        # rsi_norm
            np.clip(adx / 50.0, 0, 1),          # adx_norm
            np.clip(macd / (close * 0.001 + 1e-4), -1, 1),  # macd_norm
            np.clip(atr / (close * 0.01 + 1e-4), 0, 1),     # atr_norm
            np.clip(volume / (np.mean(volume) * 3 + 1e-4), 0, 1),  # vol_norm
        ]).astype(np.float32)

        # Generar secuencias
        valid_start = 60  # Esperar a que los indicadores se estabilicen
        count = 0
        for i in range(valid_start + SEQ_LENGTH, len(features) - PRED_HORIZON):
            seq = features[i - SEQ_LENGTH:i]
            if np.any(np.isnan(seq)) or np.any(np.isinf(seq)):
                continue
            current_close = close[i]
            future_close = close[i + PRED_HORIZON - 1]
            label = 1 if future_close > current_close else 0
            all_sequences.append(seq)
            all_labels.append(label)
            count += 1

        print(f"  {name}: {count} secuencias generadas ({len(data)} candles)")

    except Exception as e:
        print(f"  [ERROR] {name}: {e}")

print(f"\nTotal: {len(all_sequences)} secuencias de entrenamiento")
print(f"Balance: {sum(all_labels)/len(all_labels)*100:.1f}% UP / {(1-sum(all_labels)/len(all_labels))*100:.1f}% DOWN")


# ============================================================
# CELDA 4: Entrenar
# ============================================================
# ============ CELDA 4: Entrenar el modelo en GPU ============

# Convertir a tensores
X = torch.FloatTensor(np.array(all_sequences))
y = torch.FloatTensor(np.array(all_labels))

# Split train/val (80/20)
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Modelo en GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TorchLSTMGPU(input_size=6, hidden_size=128, num_layers=2, num_heads=4).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.BCELoss()
scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

# OneCycleLR
EPOCHS = 50
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.002, steps_per_epoch=len(train_loader),
    epochs=EPOCHS, pct_start=0.3, anneal_strategy='cos'
)

print(f"\nEntrenando {EPOCHS} epochs en {device}...")
print(f"Parametros: {sum(p.numel() for p in model.parameters()):,}")
print()

best_val_acc = 0
train_losses = []
val_accs = []

for epoch in range(EPOCHS):
    # Train
    model.train()
    epoch_loss = 0
    n_batches = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast('cuda'):
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    train_losses.append(avg_loss)

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            pred = model(batch_X)
            predicted = (pred > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += len(batch_y)

    val_acc = correct / total
    val_accs.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")

    if (epoch + 1) % 5 == 0 or epoch == 0:
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.1%} | Best: {best_val_acc:.1%} | LR: {lr:.2e}")

print(f"\n{'='*50}")
print(f"ENTRENAMIENTO COMPLETO")
print(f"Mejor precision: {best_val_acc:.1%}")
print(f"Modelo guardado: best_model.pt")


# ============================================================
# CELDA 5: Evaluar
# ============================================================
# ============ CELDA 5: Evaluar y visualizar ============
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
ax1.plot(train_losses, color='#00bcd4', linewidth=2)
ax1.set_title('Training Loss', fontsize=14)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('BCE Loss')
ax1.grid(True, alpha=0.3)

# Validation accuracy
ax2.plot(val_accs, color='#00c853', linewidth=2)
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
ax2.axhline(y=best_val_acc, color='#ffc107', linestyle='--', alpha=0.5, label=f'Best: {best_val_acc:.1%}')
ax2.set_title('Validation Accuracy', fontsize=14)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Grafico guardado: training_curves.png")


# ============================================================
# CELDA 6: Exportar
# ============================================================
# ============ CELDA 6: Exportar modelo para el bot local ============

# Cargar mejor modelo
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Guardar checkpoint completo compatible con lstm_predictor.py
GPU_CONFIG = {
    "hidden_size": 128,
    "num_layers": 2,
    "num_heads": 4,
    "dropout": 0.2,
    "batch_size": 64,
    "epochs_per_train": 20,
    "lr": 0.001,
    "lr_min": 1e-6,
    "max_grad_norm": 1.0,
    "weight_decay": 1e-4,
    "sequence_length": 30,
    "buffer_size": 2000,
    "use_amp": True,
}

checkpoint = {
    "model_state": model.state_dict(),
    "config": GPU_CONFIG,
    "is_trained": True,
    "total_predictions": 0,
    "correct_predictions": 0,
    "training_info": {
        "epochs": EPOCHS,
        "best_val_acc": best_val_acc,
        "final_loss": train_losses[-1],
        "pairs_trained": list(PAIRS.keys()),
        "sequences": len(all_sequences),
        "trained_on": "Google Colab GPU",
        "date": datetime.now().isoformat(),
    }
}

torch.save(checkpoint, "lstm_model_colab.pt")
print(f"Checkpoint guardado: lstm_model_colab.pt")
print(f"Tamano: {os.path.getsize('lstm_model_colab.pt') / 1024:.1f} KB")
print()
print("INSTRUCCIONES:")
print("1. Descarga lstm_model_colab.pt desde Colab")
print("2. Copialo a C:\\Dev\\ML_SuperTrend_Bot\\lstm_model.pt")
print("3. El bot lo cargara automaticamente al iniciar")

# Descargar automaticamente en Colab
try:
    from google.colab import files
    files.download("lstm_model_colab.pt")
    print("\n[OK] Descarga iniciada automaticamente!")
except:
    print("\n[INFO] Para descargar manualmente: click derecho en lstm_model_colab.pt \u2192 Download")

