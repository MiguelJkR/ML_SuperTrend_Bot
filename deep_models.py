"""
ML SuperTrend v51 - Deep Learning Models Avanzados
====================================================
Modelos de deep learning adicionales para mejorar predicción.

Contiene:
  1. Variable Selection Network (VSN) — del TFT de Google
  2. Gated Residual Network (GRN) — backbone del TFT
  3. Contrastive Learning (TS2Vec) — representaciones sin labels
  4. Variational Autoencoder (VAE) — anomaly detection
  5. Knowledge Distillation — teacher-student
  6. MAML (Meta-Learning) — adaptación rápida a regímenes
  7. Graph Attention Network — cross-asset dependencies

Papers:
  - Lim et al. (2021) — Temporal Fusion Transformer
  - Yue et al. (2022) — TS2Vec
  - Kingma & Welling (2014) — VAE
  - Hinton et al. (2015) — Knowledge Distillation
  - Finn et al. (2017) — MAML
  - Xu et al. (2020) — Graph Networks for Finance

Uso:
    from deep_models import (
        VariableSelectionNetwork, ContrastiveLearner,
        MarketVAE, KnowledgeDistiller, MAMLTrainer,
        CrossAssetGNN
    )
"""

import logging
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from collections import deque
from copy import deepcopy

logger = logging.getLogger(__name__)

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    pass


# =====================================================================
# 1. GATED RESIDUAL NETWORK (GRN) — Backbone del TFT
# =====================================================================

if TORCH_AVAILABLE:
    class GatedResidualNetwork(nn.Module):
        """
        GRN del Temporal Fusion Transformer.
        \u03b7 = LayerNorm(a + GLU(W_1\u00b7a + W_2\u00b7context + b))
        donde a = ELU(W_3\u00b7x + b_3)

        Permite flujo de informaci\u00f3n selectivo con gating.
        """

        def __init__(self, input_dim: int, hidden_dim: int = 64,
                     output_dim: int = None, context_dim: int = None, dropout: float = 0.1):
            super().__init__()
            output_dim = output_dim or input_dim

            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.elu = nn.ELU()
            self.fc2 = nn.Linear(hidden_dim, output_dim * 2)  # *2 for GLU
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(output_dim)

            self.context_fc = None
            if context_dim is not None:
                self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)

            self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

        def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            residual = self.skip(x) if self.skip else x

            a = self.elu(self.fc1(x))
            if self.context_fc is not None and context is not None:
                a = a + self.context_fc(context)

            gate_input = self.fc2(a)
            gate_input = self.dropout(gate_input)

            # GLU: split in half, sigmoid gate \u00d7 value
            value, gate = gate_input.chunk(2, dim=-1)
            gated = torch.sigmoid(gate) * value

            return self.layer_norm(residual + gated)


    # =====================================================================
    # 2. VARIABLE SELECTION NETWORK (VSN) \u2014 TFT Feature Selection
    # =====================================================================

    class VariableSelectionNetwork(nn.Module):
        """
        VSN del TFT: aprende qu\u00e9 features son importantes y cu\u00e1les ignorar.
        VSN(x) = Softmax(GRN_v(flatten(x))) \u2299 [GRN_1(x_1), ..., GRN_n(x_n)]

        Cada feature pasa por su propio GRN, y un GRN global decide
        el peso de cada feature con softmax gating.
        """

        def __init__(self, n_features: int, hidden_dim: int = 32,
                     context_dim: int = None, dropout: float = 0.1):
            super().__init__()
            self.n_features = n_features

            # GRN por cada feature
            self.feature_grns = nn.ModuleList([
                GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout=dropout)
                for _ in range(n_features)
            ])

            # GRN global para selecci\u00f3n
            self.selection_grn = GatedResidualNetwork(
                n_features, hidden_dim, n_features,
                context_dim=context_dim, dropout=dropout
            )

            self.softmax = nn.Softmax(dim=-1)
            self.last_weights = None

        def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            x: (batch, seq_len, n_features) o (batch, n_features)
            Returns: (selected_output, selection_weights)
            """
            has_seq = x.dim() == 3

            if has_seq:
                batch, seq_len, n_feat = x.shape
                x_flat = x.reshape(batch * seq_len, n_feat)
            else:
                x_flat = x

            # Procesar cada feature por su GRN
            processed = []
            for i in range(self.n_features):
                feat_i = x_flat[:, i:i+1]  # (batch, 1)
                processed.append(self.feature_grns[i](feat_i))  # (batch, hidden)

            processed = torch.stack(processed, dim=1)  # (batch, n_features, hidden)

            # Selection weights
            selection_input = x_flat  # (batch, n_features)
            ctx = context.reshape(-1, context.shape[-1]) if context is not None and has_seq else context
            selection_weights = self.softmax(
                self.selection_grn(selection_input, ctx)
            )  # (batch, n_features)

            self.last_weights = selection_weights.detach()

            # Weighted combination
            # (batch, n_features, 1) \u00d7 (batch, n_features, hidden)
            weighted = selection_weights.unsqueeze(-1) * processed
            output = weighted.sum(dim=1)  # (batch, hidden)

            if has_seq:
                output = output.reshape(batch, seq_len, -1)
                selection_weights = selection_weights.reshape(batch, seq_len, -1)

            return output, selection_weights

        def get_feature_importance(self) -> Optional[np.ndarray]:
            """Importancia promedio de cada feature."""
            if self.last_weights is None:
                return None
            return self.last_weights.mean(dim=0).cpu().numpy()


    # =====================================================================
    # 3. CONTRASTIVE LEARNING (TS2Vec adaptado)
    # =====================================================================

    class ContrastiveLearner(nn.Module):
        """
        Contrastive learning para series temporales.
        Aprende representaciones \u00fatiles SIN labels.

        Idea: dos ventanas del mismo r\u00e9gimen \u2192 representaciones similares
              dos ventanas de r\u00e9gimenes diferentes \u2192 representaciones distintas

        Loss: InfoNCE
        L = -log(exp(sim(z_i, z_j)/\u03c4) / \u03a3 exp(sim(z_i, z_k)/\u03c4))
        """

        def __init__(self, input_dim: int = 6, hidden_dim: int = 64,
                     output_dim: int = 32, temperature: float = 0.07):
            super().__init__()
            self.temperature = temperature

            # Encoder: secuencia \u2192 representaci\u00f3n
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

            # Projection head (se descarta despu\u00e9s de pre-training)
            self.projector = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

            self.hidden_dim = hidden_dim

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            """Codificar secuencia a representaci\u00f3n fija."""
            # x: (batch, seq_len, input_dim)
            # Promedio temporal
            x_avg = x.mean(dim=1)  # (batch, input_dim)
            return self.encoder(x_avg)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward para contrastive training."""
            h = self.encode(x)
            z = self.projector(h)
            return F.normalize(z, dim=-1)

        def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
            """
            InfoNCE loss entre dos vistas del mismo dato.
            z1, z2: (batch, output_dim) \u2014 representaciones normalizadas
            """
            batch_size = z1.shape[0]

            # Similaridad coseno
            sim_matrix = torch.mm(z1, z2.t()) / self.temperature  # (batch, batch)

            # Labels: diagonal = positivos
            labels = torch.arange(batch_size, device=z1.device)

            # Cross-entropy en ambas direcciones
            loss_12 = F.cross_entropy(sim_matrix, labels)
            loss_21 = F.cross_entropy(sim_matrix.t(), labels)

            return (loss_12 + loss_21) / 2

        def pretrain_step(self, x_batch: torch.Tensor) -> float:
            """
            Un paso de pre-entrenamiento contrastivo.
            Genera dos vistas augmentadas del mismo batch.
            """
            # Vista 1: original
            z1 = self(x_batch)

            # Vista 2: jittering + masking
            noise = torch.randn_like(x_batch) * 0.03
            x_aug = x_batch + noise
            # Random masking (10% de features)
            mask = torch.rand_like(x_batch) > 0.1
            x_aug = x_aug * mask.float()
            z2 = self(x_aug)

            loss = self.contrastive_loss(z1, z2)
            return loss


    # =====================================================================
    # 4. VARIATIONAL AUTOENCODER (VAE) \u2014 Anomaly Detection
    # =====================================================================

    class MarketVAE(nn.Module):
        """
        VAE para detecci\u00f3n de anomal\u00edas en el mercado.
        Comprime 30 bars \u00d7 6 features \u2192 vector latente de 8-16 dims.

        Si la reconstrucci\u00f3n es mala \u2192 mercado an\u00f3malo \u2192 no operar.

        L = E[log p(x|z)] - KL(q(z|x) || p(z))
        """

        def __init__(self, input_dim: int = 6, seq_len: int = 30,
                     hidden_dim: int = 64, latent_dim: int = 12):
            super().__init__()
            self.input_dim = input_dim
            self.seq_len = seq_len
            self.latent_dim = latent_dim
            flat_dim = input_dim * seq_len

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(flat_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, flat_dim),
            )

            self.last_recon_error = 0.0
            self.anomaly_threshold = 0.5  # Se calibra con datos normales

        def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Encode to latent space."""
            flat = x.reshape(x.shape[0], -1)
            h = self.encoder(flat)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            """Reparameterization trick: z = \u03bc + \u03c3\u00b7\u03b5"""
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
            return mu

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            """Decode from latent space."""
            flat = self.decoder(z)
            return flat.reshape(-1, self.seq_len, self.input_dim)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            x: (batch, seq_len, input_dim)
            Returns: (reconstruction, mu, logvar)
            """
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            return recon, mu, logvar

        def loss_function(self, x: torch.Tensor, recon: torch.Tensor,
                         mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            """ELBO loss = Reconstruction + KL divergence."""
            recon_loss = F.mse_loss(recon, x, reduction='mean')
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + 0.1 * kl_loss

        def is_anomaly(self, x: torch.Tensor) -> Tuple[bool, float]:
            """
            Detectar si el mercado actual es an\u00f3malo.
            Returns: (is_anomaly, reconstruction_error)
            """
            self.eval()
            with torch.no_grad():
                recon, mu, logvar = self(x)
                error = F.mse_loss(recon, x).item()
                self.last_recon_error = error
            return error > self.anomaly_threshold, error

        def calibrate_threshold(self, normal_data: torch.Tensor, percentile: float = 95):
            """Calibrar threshold con datos normales."""
            self.eval()
            errors = []
            with torch.no_grad():
                for i in range(len(normal_data)):
                    x = normal_data[i:i+1]
                    recon, _, _ = self(x)
                    error = F.mse_loss(recon, x).item()
                    errors.append(error)
            self.anomaly_threshold = float(np.percentile(errors, percentile))
            logger.info(f"VAE threshold calibrated: {self.anomaly_threshold:.4f} (p{percentile})")


    # =====================================================================
    # 5. KNOWLEDGE DISTILLATION \u2014 Teacher \u2192 Student
    # =====================================================================

    class KnowledgeDistiller:
        """
        Destila conocimiento de un modelo grande (teacher) a uno peque\u00f1o (student).
        L = \u03b1\u00b7CE(y, \u03c3(z_s)) + (1-\u03b1)\u00b7KL(\u03c3(z_t/T), \u03c3(z_s/T))

        El student aprende las "soft probabilities" del teacher.
        """

        def __init__(self, teacher: nn.Module, student: nn.Module,
                     temperature: float = 4.0, alpha: float = 0.3):
            self.teacher = teacher
            self.student = student
            self.temperature = temperature
            self.alpha = alpha
            self.distill_losses = []

        def distillation_loss(self, student_logits: torch.Tensor,
                              teacher_logits: torch.Tensor,
                              labels: torch.Tensor) -> torch.Tensor:
            """
            Combined distillation loss.
            """
            T = self.temperature

            # Soft targets from teacher
            soft_teacher = torch.sigmoid(teacher_logits / T)
            soft_student = torch.sigmoid(student_logits / T)

            # KL divergence on soft probabilities
            kl_loss = F.mse_loss(soft_student, soft_teacher) * (T ** 2)

            # Hard label loss
            hard_loss = F.binary_cross_entropy(
                torch.sigmoid(student_logits), labels
            )

            total = self.alpha * hard_loss + (1 - self.alpha) * kl_loss
            return total

        def train_student_epoch(self, dataloader, optimizer, device='cpu') -> float:
            """Train student for one epoch with teacher guidance."""
            self.teacher.eval()
            self.student.train()
            total_loss = 0
            n_batches = 0

            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                # Teacher predictions (no gradient)
                with torch.no_grad():
                    teacher_out = self.teacher(batch_X)

                # Student predictions
                student_out = self.student(batch_X)

                loss = self.distillation_loss(student_out, teacher_out, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            self.distill_losses.append(avg_loss)
            return avg_loss

        def get_status(self) -> Dict:
            return {
                "temperature": self.temperature,
                "alpha": self.alpha,
                "n_epochs_trained": len(self.distill_losses),
                "last_loss": round(self.distill_losses[-1], 4) if self.distill_losses else None,
                "teacher_params": sum(p.numel() for p in self.teacher.parameters()),
                "student_params": sum(p.numel() for p in self.student.parameters()),
            }


    # =====================================================================
    # 6. MAML \u2014 Model-Agnostic Meta-Learning
    # =====================================================================

    class MAMLTrainer:
        """
        Meta-Learning: el modelo aprende par\u00e1metros iniciales que se
        adaptan r\u00e1pidamente a nuevos reg\u00edmenes en 2-3 gradient steps.

        \u03b8* = \u03b8 - \u03b1\u00b7\u2207_\u03b8 L(f_{\u03b8 - \u03b2\u00b7\u2207_\u03b8 L_task})

        Entrenamiento:
          1. Samplear "tareas" (diferentes pares, sesiones, reg\u00edmenes)
          2. Inner loop: adaptar a cada tarea con \u03b2
          3. Outer loop: actualizar \u03b8 para que la adaptaci\u00f3n funcione mejor
        """

        def __init__(self, model: nn.Module, inner_lr: float = 0.01,
                     outer_lr: float = 0.001, inner_steps: int = 3):
            self.model = model
            self.inner_lr = inner_lr
            self.outer_lr = outer_lr
            self.inner_steps = inner_steps
            self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
            self.meta_losses = []

        def inner_loop(self, support_x: torch.Tensor, support_y: torch.Tensor) -> nn.Module:
            """
            Inner loop: adaptar modelo a una tarea espec\u00edfica.
            Returns: modelo adaptado (copia temporal).
            """
            adapted_model = deepcopy(self.model)
            inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

            for _ in range(self.inner_steps):
                pred = adapted_model(support_x)
                loss = F.binary_cross_entropy(pred, support_y)
                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()

            return adapted_model

        def meta_train_step(self, tasks: List[Tuple[torch.Tensor, torch.Tensor,
                                                      torch.Tensor, torch.Tensor]]) -> float:
            """
            Un paso de meta-entrenamiento.

            Args:
                tasks: lista de (support_x, support_y, query_x, query_y) por tarea
                       support = datos para adaptar, query = datos para evaluar

            Returns:
                meta loss promedio
            """
            self.meta_optimizer.zero_grad()
            meta_loss = torch.tensor(0.0, requires_grad=True)

            for support_x, support_y, query_x, query_y in tasks:
                # Inner loop: adaptar a esta tarea
                adapted = self.inner_loop(support_x, support_y)

                # Evaluar en query set
                query_pred = adapted(query_x)
                task_loss = F.binary_cross_entropy(query_pred, query_y)
                meta_loss = meta_loss + task_loss

            meta_loss = meta_loss / len(tasks)

            # Outer loop update (meta-gradiente)
            # Nota: esto usa first-order MAML (sin second-order gradients)
            # para eficiencia computacional
            meta_loss.backward()
            self.meta_optimizer.step()

            self.meta_losses.append(meta_loss.item())
            return meta_loss.item()

        def create_tasks_from_data(self, X: torch.Tensor, y: torch.Tensor,
                                    n_tasks: int = 4, support_size: int = 20) -> List:
            """Crear tareas de meta-learning dividiendo datos por segmentos temporales."""
            tasks = []
            n = len(X)
            segment_size = n // n_tasks

            for i in range(n_tasks):
                start = i * segment_size
                end = min(start + segment_size, n)
                segment_x = X[start:end]
                segment_y = y[start:end]

                if len(segment_x) < support_size * 2:
                    continue

                # Split en support/query
                support_x = segment_x[:support_size]
                support_y = segment_y[:support_size]
                query_x = segment_x[support_size:]
                query_y = segment_y[support_size:]

                tasks.append((support_x, support_y, query_x, query_y))

            return tasks

        def get_status(self) -> Dict:
            return {
                "inner_lr": self.inner_lr,
                "outer_lr": self.outer_lr,
                "inner_steps": self.inner_steps,
                "meta_epochs": len(self.meta_losses),
                "last_meta_loss": round(self.meta_losses[-1], 4) if self.meta_losses else None,
            }


    # =====================================================================
    # 7. GRAPH ATTENTION NETWORK \u2014 Cross-Asset Dependencies
    # =====================================================================

    class CrossAssetGNN(nn.Module):
        """
        Graph Attention Network para modelar dependencias entre pares de divisas.

        Nodos: EUR/USD, GBP/USD, USD/JPY, etc. (cada par es un nodo)
        Edges: correlaciones entre pares (din\u00e1micas)

        h_v = \u03c3(\u03a3 \u03b1_uv W h_u) \u2014 atenci\u00f3n sobre vecinos

        Detecta divergencias y oportunidades entre pares.
        """

        def __init__(self, n_assets: int = 8, feature_dim: int = 6,
                     hidden_dim: int = 32, n_heads: int = 2):
            super().__init__()
            self.n_assets = n_assets
            self.feature_dim = feature_dim
            self.hidden_dim = hidden_dim
            self.n_heads = n_heads

            # Node feature transformation
            self.W = nn.Linear(feature_dim, hidden_dim * n_heads, bias=False)

            # Attention mechanism
            self.attention = nn.Parameter(torch.zeros(n_heads, 2 * hidden_dim))
            nn.init.xavier_uniform_(self.attention.unsqueeze(0))

            self.leaky_relu = nn.LeakyReLU(0.2)
            self.softmax = nn.Softmax(dim=-1)

            # Output
            self.output_fc = nn.Linear(hidden_dim * n_heads, hidden_dim)

            self.last_attention_matrix = None
            self.asset_names = [
                "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
                "AUD_USD", "NZD_USD", "USD_CAD", "XAU_USD"
            ][:n_assets]

        def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
            """
            x: (batch, n_assets, feature_dim) \u2014 features de cada par
            adj: (n_assets, n_assets) \u2014 adjacency matrix (correlaciones)
                 Si None, usa fully-connected

            Returns: (batch, n_assets, hidden_dim) \u2014 representaciones enriquecidas
            """
            batch_size = x.shape[0]

            # Transform features: (batch, n_assets, hidden * n_heads)
            h = self.W(x)
            h = h.view(batch_size, self.n_assets, self.n_heads, self.hidden_dim)
            # (batch, n_assets, n_heads, hidden)

            # Compute attention for each pair of nodes
            attn_scores = torch.zeros(batch_size, self.n_heads, self.n_assets, self.n_assets,
                                       device=x.device)

            for i in range(self.n_assets):
                for j in range(self.n_assets):
                    # Concatenate node i and node j features
                    concat_ij = torch.cat([h[:, i], h[:, j]], dim=-1)  # (batch, n_heads, 2*hidden)
                    # Dot with attention vector
                    score = (concat_ij * self.attention.unsqueeze(0)).sum(dim=-1)  # (batch, n_heads)
                    attn_scores[:, :, i, j] = self.leaky_relu(score)

            # Mask with adjacency if provided
            if adj is not None:
                mask = adj.unsqueeze(0).unsqueeze(0)  # (1, 1, n, n)
                attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

            # Softmax per source node
            attn_weights = self.softmax(attn_scores)  # (batch, n_heads, n_assets, n_assets)
            self.last_attention_matrix = attn_weights.detach()

            # Aggregate neighbor features
            # h: (batch, n_assets, n_heads, hidden) \u2192 (batch, n_heads, n_assets, hidden)
            h_perm = h.permute(0, 2, 1, 3)
            out = torch.matmul(attn_weights, h_perm)  # (batch, n_heads, n_assets, hidden)

            # Concatenate heads
            out = out.permute(0, 2, 1, 3).reshape(batch_size, self.n_assets, -1)
            out = self.output_fc(out)  # (batch, n_assets, hidden_dim)

            return out

        def get_cross_asset_scores(self) -> Optional[Dict]:
            """Obtener scores de interacci\u00f3n entre activos."""
            if self.last_attention_matrix is None:
                return None

            # Promediar sobre batch y heads
            avg_attn = self.last_attention_matrix[0].mean(dim=0).cpu().numpy()

            interactions = {}
            for i in range(min(self.n_assets, len(self.asset_names))):
                for j in range(min(self.n_assets, len(self.asset_names))):
                    if i != j and avg_attn[i, j] > 0.15:  # Solo interacciones significativas
                        key = f"{self.asset_names[i]}\u2192{self.asset_names[j]}"
                        interactions[key] = round(float(avg_attn[i, j]), 4)

            return interactions

        def get_status(self) -> Dict:
            return {
                "n_assets": self.n_assets,
                "n_heads": self.n_heads,
                "hidden_dim": self.hidden_dim,
                "interactions": self.get_cross_asset_scores(),
            }

else:
    # Placeholder classes when PyTorch not available
    class VariableSelectionNetwork:
        def __init__(self, *args, **kwargs): pass
    class ContrastiveLearner:
        def __init__(self, *args, **kwargs): pass
    class MarketVAE:
        def __init__(self, *args, **kwargs): pass
    class KnowledgeDistiller:
        def __init__(self, *args, **kwargs): pass
    class MAMLTrainer:
        def __init__(self, *args, **kwargs): pass
    class CrossAssetGNN:
        def __init__(self, *args, **kwargs): pass
    class GatedResidualNetwork:
        def __init__(self, *args, **kwargs): pass
