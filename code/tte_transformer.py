import torch
import torch.nn as nn
from config import config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, T_max, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class TransformerTTE(nn.Module):
    def __init__(
        self,
        input_dim_context,
        d_model,
        nhead,
        num_layers,
        dropout,
        dropout_time,
        max_seq_len,
        dow_embedding_dim=8,
        num_dow=2,
        use_positional_encoding=True,
        use_alpha_fusion=True,
        use_abs_time_scale=True,
        use_time_features=True,
        use_context_features=True,
        use_DoW=True
    ):
        super(TransformerTTE, self).__init__()

        # Store toggles
        self.use_positional_encoding = use_positional_encoding
        self.use_alpha_fusion = use_alpha_fusion
        self.use_abs_time_scale = use_abs_time_scale
        self.use_time_features = use_time_features
        self.use_context_features = use_context_features
        self.use_DoW = use_DoW

        # Day-of-week embedding
        self.dow_embedding = nn.Embedding(num_dow, dow_embedding_dim)

        # Absolute time projection
        self.abs_time_proj = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.LayerNorm(8)
        )
        self.abs_time_dropout = nn.Dropout(p=dropout_time)
        self.abs_time_linear = nn.Linear(8, d_model)
        if self.use_abs_time_scale:  
            self.abs_time_scale = nn.Parameter(torch.tensor(1.0))  # Learnable scaling
        else:
            self.register_buffer('abs_time_scale', torch.tensor(1.0))

        self.alpha_param = nn.Parameter(torch.tensor(0.5))  # Fusion weight

        # Context projection
        #self.context_dow_dim = input_dim_context + dow_embedding_dim
        context_input_dim = input_dim_context
        if self.use_DoW:
            context_input_dim += dow_embedding_dim
        
        #self.input_proj = nn.Linear(self.context_dow_dim, d_model)
        self.input_proj = nn.Linear(context_input_dim, d_model)
        
        self.input_layernorm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        self.pe_dropout = nn.Dropout(dropout)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x_context, abs_time_scalar, dow_idx):
        B, T, _ = x_context.shape

        # --- Feature Preparation ---
        context_dow_proj = torch.zeros(B, T, self.input_proj.out_features, device=x_context.device)
        if self.use_context_features:
            # DoW embedding
            if self.use_DoW:
                dow_expand = dow_idx.unsqueeze(1).expand(-1, T)
                dow_embed = self.dow_embedding(dow_expand)
                context_input = torch.cat([x_context, dow_embed], dim=-1)
            else:
                context_input = x_context

            context_dow_proj = self.input_proj(context_input)
            context_dow_proj = self.input_layernorm(context_dow_proj)

        abs_time_embed = torch.zeros_like(context_dow_proj)
        if self.use_time_features:
            _abs_time_embed = self.abs_time_proj(abs_time_scalar)
            _abs_time_embed = self.abs_time_dropout(_abs_time_embed)
            _abs_time_embed = self.abs_time_linear(_abs_time_embed)
            if self.use_abs_time_scale:
                _abs_time_embed = self.abs_time_scale * _abs_time_embed
            abs_time_embed = _abs_time_embed
            
        # --- Fusion Strategy ---
        if self.use_context_features and self.use_time_features:
            if self.use_alpha_fusion:
                alpha = torch.sigmoid(self.alpha_param)
                x = alpha * context_dow_proj + (1 - alpha) * abs_time_embed
            else:
                x = context_dow_proj + abs_time_embed
        elif self.use_context_features:
            x = context_dow_proj
        elif self.use_time_features:
            x = abs_time_embed
        else:
            x = torch.zeros_like(context_dow_proj)

        x = self.input_dropout(x)

        # Positional encoding 
        if self.use_positional_encoding:
            x = self.pos_encoder(x)
        x = self.pe_dropout(x)

        # Transformer encoder
        z = self.transformer_encoder(x)

        # Output q(t|x)
        out = self.output_layer(z).squeeze(-1)

        return out