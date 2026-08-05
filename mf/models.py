"""
mf/models.py — Architectures de prévision multi-horizon (PyTorch).

Interface commune
-----------------
  entrée  x : (B, L, F)
  sortie    : (B, H)      si n_quantiles == 1  (prévision ponctuelle)
              (B, H, Q)   si n_quantiles  > 1  (prévision probabiliste)

Modèles : TCN, BiLSTM, CNNBiLSTMAttention, TFTLite (TFT compact).
Tous partagent une tête linéaire produisant H × Q valeurs.
"""
import torch
import torch.nn as nn


class _Head(nn.Module):
    """Tête linéaire : représentation (B, D) → (B, H) ou (B, H, Q)."""
    def __init__(self, in_dim, horizon, n_quantiles=1, hidden=128, p=0.1):
        super().__init__()
        self.H, self.Q = horizon, n_quantiles
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, horizon * n_quantiles))

    def forward(self, z):
        out = self.net(z)
        return out.view(-1, self.H, self.Q) if self.Q > 1 else out


# ── TCN (convolutions causales dilatées) ──────────────────────────────────────

class _Chomp(nn.Module):
    def __init__(self, c): super().__init__(); self.c = c
    def forward(self, x): return x[:, :, :-self.c].contiguous() if self.c else x


class _TempBlock(nn.Module):
    def __init__(self, ci, co, k, d, p):
        super().__init__()
        pad = (k - 1) * d
        self.net = nn.Sequential(
            nn.Conv1d(ci, co, k, padding=pad, dilation=d), _Chomp(pad),
            nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(co, co, k, padding=pad, dilation=d), _Chomp(pad),
            nn.ReLU(), nn.Dropout(p))
        self.down = nn.Conv1d(ci, co, 1) if ci != co else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, n_features, horizon, n_quantiles=1,
                 channels=(64, 64, 64), kernel=3, dropout=0.1, head_hidden=128):
        super().__init__()
        layers, ci = [], n_features
        for i, co in enumerate(channels):
            layers.append(_TempBlock(ci, co, kernel, 2 ** i, dropout)); ci = co
        self.tcn = nn.Sequential(*layers)
        self.head = _Head(ci, horizon, n_quantiles, head_hidden, dropout)

    def forward(self, x):                      # x (B,L,F)
        z = self.tcn(x.transpose(1, 2))        # (B,C,L)
        return self.head(z[:, :, -1])          # dernier pas temporel


# ── BiLSTM ────────────────────────────────────────────────────────────────────

class BiLSTM(nn.Module):
    def __init__(self, n_features, horizon, n_quantiles=1, hidden=128,
                 layers=2, dropout=0.1, head_hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = _Head(hidden * 2, horizon, n_quantiles, head_hidden, dropout)

    def forward(self, x):                      # x (B,L,F)
        out, _ = self.lstm(x)                  # (B,L,2H)
        return self.head(out[:, -1, :])        # dernier état temporel


# ── CNN-BiLSTM-Attention ──────────────────────────────────────────────────────

class _AdditiveAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, dim); self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, h):                      # h (B,L,D)
        score = self.v(torch.tanh(self.w(h))).squeeze(-1)   # (B,L)
        a = torch.softmax(score, dim=1).unsqueeze(-1)       # (B,L,1)
        return (a * h).sum(1), a                            # contexte (B,D)


class CNNBiLSTMAttention(nn.Module):
    def __init__(self, n_features, horizon, n_quantiles=1, cnn_ch=64,
                 hidden=128, layers=1, kernel=3, dropout=0.1, head_hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_ch, kernel, padding=kernel // 2),
            nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(cnn_ch, hidden, layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.attn = _AdditiveAttention(hidden * 2)
        self.head = _Head(hidden * 2, horizon, n_quantiles, head_hidden, dropout)

    def forward(self, x):                      # x (B,L,F)
        z = self.cnn(x.transpose(1, 2)).transpose(1, 2)     # (B,L,C)
        h, _ = self.lstm(z)                                 # (B,L,2H)
        ctx, _ = self.attn(h)
        return self.head(ctx)


# ── TFT compact (Temporal Fusion Transformer, variante légère) ────────────────

class _GRN(nn.Module):
    """Gated Residual Network — bloc de base du TFT (gating + résidu + norm)."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(dim, dim)
        self.fc2  = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):                                  # x (..., dim)
        h = self.drop(self.fc2(nn.functional.elu(self.fc1(x))))
        g = torch.sigmoid(self.gate(x))
        return self.norm(x + g * h)


class TFTLite(nn.Module):
    """
    Variante compacte du Temporal Fusion Transformer :
      projection des features → encodeur LSTM → attention multi-tête
      interprétable (self-attention) + Gated Residual Networks → tête quantile.
    Interface commune : (B,L,F) → (B,H) ou (B,H,Q).
    NB : pour le TFT canonique complet (sélection de variables, covariables
    statiques/connues-du-futur séparées), utiliser pytorch-forecasting ;
    un adaptateur peut être fourni sur demande.
    """
    def __init__(self, n_features, horizon, n_quantiles=1, d_model=128,
                 lstm_layers=1, n_heads=4, dropout=0.1, head_hidden=128):
        super().__init__()
        self.input    = nn.Linear(n_features, d_model)
        self.enc      = nn.LSTM(d_model, d_model, lstm_layers,
                                batch_first=True,
                                dropout=dropout if lstm_layers > 1 else 0.0)
        self.enc_grn  = _GRN(d_model, dropout)
        self.attn     = nn.MultiheadAttention(d_model, n_heads,
                                              dropout=dropout, batch_first=True)
        self.norm     = nn.LayerNorm(d_model)
        self.post_grn = _GRN(d_model, dropout)
        self.head     = _Head(d_model, horizon, n_quantiles, head_hidden, dropout)

    def forward(self, x):                      # x (B,L,F)
        z = self.input(x)                      # (B,L,d)
        h, _ = self.enc(z)                     # (B,L,d)
        h = self.enc_grn(h)
        a, _ = self.attn(h, h, h)              # self-attention (B,L,d)
        h = self.post_grn(self.norm(h + a))    # résidu + gating
        return self.head(h[:, -1, :])          # contexte au dernier pas


# ── Fabrique ──────────────────────────────────────────────────────────────────

def build_model(name, n_features, horizon, n_quantiles=1, **kw):
    name = name.lower()
    if name == "tcn":
        return TCN(n_features, horizon, n_quantiles, **kw)
    if name == "bilstm":
        return BiLSTM(n_features, horizon, n_quantiles, **kw)
    if name in ("cnnbilstmattention", "cnn_bilstm_attn", "hybrid"):
        return CNNBiLSTMAttention(n_features, horizon, n_quantiles, **kw)
    if name in ("tft", "tftlite"):
        return TFTLite(n_features, horizon, n_quantiles, **kw)
    raise ValueError(f"Modèle inconnu : {name} "
                     f"(tcn | bilstm | cnnbilstmattention | tft)")
