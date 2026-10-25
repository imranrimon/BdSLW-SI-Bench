"""SPOTER-family pose transformers (graph-free), including the BdSL-SPOTER variant."""
import copy
from typing import Optional

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    SPOTER-family pose transformers (graph-free), ported to the repo contract
    Model(num_class, num_point, num_person, graph, graph_args, in_channels) with
    input (N, C, T, V, M). Each frame's V*C coordinates are flattened into one token.

    variant='bdsl_spoter' (default) -- BdSL-SPOTER (Azad & Rahman, ISVC 2025;
        github.com/sayad-dot/BDSLW_SPOTER, `ProductionSPOTER`): input projection
        (Linear+LayerNorm+Dropout) to d_model=108, prepended CLS token, learnable
        positional encoding, encoder-only nn.TransformerEncoder (9 heads, FFN 512,
        GELU, dropout 0.15), CLS -> 3-layer GELU MLP head. The paper specifies
        4 encoder layers; the released notebook uses 8 (num_layers is a knob).
        Their input is 54 landmarks x 2D = 108 dims; ours is 27 x 3 = 81, so the
        input projection maps 81 -> 108 (d_model kept at the paper's value).

    variant='spoter' -- original SPOTER (Bohacek & Hruz, WACV-W 2022;
        github.com/matyasbohacek/spoter): nn.Transformer with d_model equal to the
        flattened input dim (here 81 = 27x3, 9 heads), 6 encoder + 6 decoder layers,
        decoder layers without self-attention, a single learnable class query, and
        the original's single learnable positional vector broadcast over frames.
    """

    def __init__(self, num_class=60, num_point=27, num_person=1, graph=None, graph_args=dict(),
                 in_channels=3, variant='bdsl_spoter', d_model=108, num_heads=9, num_layers=4,
                 dim_feedforward=512, dropout=0.15, max_len=300,
                 num_encoder_layers=6, num_decoder_layers=6, spoter_ffn=2048, spoter_dropout=0.1):
        super(Model, self).__init__()
        self.variant = variant
        self.input_dim = num_point * in_channels  # 27 * 3 = 81

        if variant == 'bdsl_spoter':
            self.input_projection = nn.Sequential(
                nn.Linear(self.input_dim, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout * 0.5))
            self.positional_encoding = nn.Parameter(torch.randn(max_len + 1, d_model) * 0.1)
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                               dim_feedforward=dim_feedforward, dropout=dropout,
                                               activation='gelu', batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.class_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
                nn.Linear(d_model // 2, num_class))
            self.apply(self._init_weights)
        elif variant == 'spoter':
            hidden = self.input_dim
            assert hidden % num_heads == 0, "SPOTER d_model (V*C) must be divisible by num_heads"
            self.row_embed = nn.Parameter(torch.rand(50, hidden))
            self.pos = nn.Parameter(self.row_embed[0].detach().clone().view(1, 1, hidden))
            self.class_query = nn.Parameter(torch.rand(1, hidden))
            self.transformer = nn.Transformer(hidden, num_heads, num_encoder_layers, num_decoder_layers)
            dec = _SPOTERDecoderLayer(hidden, num_heads, spoter_ffn, spoter_dropout, 'relu')
            self.transformer.decoder.layers = nn.ModuleList(
                [copy.deepcopy(dec) for _ in range(num_decoder_layers)])
            self.linear_class = nn.Linear(hidden, num_class)
        else:
            raise ValueError(f"unknown SPOTER variant {variant!r}")

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x, keep_prob=1.0):
        # x: (N, C, T, V, M) -> (N, T, V*C)
        N, C, T, V, M = x.size()
        x = x.mean(dim=4) if M > 1 else x.squeeze(4)
        x = x.permute(0, 2, 3, 1).contiguous().view(N, T, V * C)

        if self.variant == 'bdsl_spoter':
            x = self.input_projection(x)
            x = torch.cat([self.class_token.expand(N, -1, -1), x], dim=1)
            x = x + self.positional_encoding[:T + 1].unsqueeze(0)
            x = self.transformer_encoder(x)
            return self.classifier(x[:, 0])

        # original SPOTER: sequence-first nn.Transformer, class query as the only target token
        src = (x + self.pos).transpose(0, 1)                       # (T, N, D)
        tgt = self.class_query.unsqueeze(1).expand(-1, N, -1)      # (1, N, D)
        h = self.transformer(src, tgt).squeeze(0)                  # (N, D)
        return self.linear_class(h)


class _NoSelfAttn(nn.Module):
    batch_first = False


class _SPOTERDecoderLayer(nn.TransformerDecoderLayer):
    """SPOTER's decoder layer: TransformerDecoderLayer without the self-attention block."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(_SPOTERDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        # Drop the self-attention weights; keep a parameter-free stub because
        # nn.TransformerDecoder reads layers[0].self_attn.batch_first.
        self.self_attn = _NoSelfAttn()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return self.norm3(tgt)
