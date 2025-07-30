# ---- tide_lite.py ----------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseLightning  # your class


# ──────────────────────────────────────────────────────────────────────
class _Residual(nn.Module):
    def __init__(self, inp, out, hidden, dropout=0.1, layer_norm=True):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, out), nn.Dropout(dropout))
        self.skip = nn.Linear(inp, out)
        self.norm = nn.LayerNorm(out) if layer_norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        return self.norm(self.block(x) + self.skip(x))


# ──────────────────────────────────────────────────────────────────────
class TiDELite(BaseLightning):
    """
    A minimal TiDE implementation adapted to the batch‑dict  produced
    by your TimeSeriesDataset.  Assumes:

        batch = {
            "y_hist":        [B,L]               # target history
            "X_past":        [B,L,Dp]            # past covariates     (optional)
            "X_futr_hist":   [B,L,Df]            # hist.  future covs  (optional)
            "X_futr":        [B,H,Df]            # future covariates   (optional)
            "static":        dict / tensor       # static (ignored here)
        }

    Only **point forecasts** (nr_params = 1) are produced.
    """

    # ---- constructor only stores hyper‑params ----------------------- #
    def __init__(
        self,
        input_length: int,
        horizons: int | list[int],
        hidden_size: int = 128,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        decoder_width: int = 64,
        temporal_hidden: int = 128,
        past_categorical_embedding_sizes: dict[str, int | tuple[int, int]] | None = None,
        future_categorical_embedding_sizes: dict[str, int | tuple[int, int]] | None = None,
        static_categorical_embedding_sizes: dict[str, int | tuple[int, int]] | None = None,
        dropout: float = 0.1,
        layer_norm: bool = True,
        use_static_covariates: bool = True,
        **kwargs,
    ):
        super().__init__(input_length, horizons, **kwargs)

        # placeholders – real modules are built in setup()
        self.encoders: nn.Module = nn.Identity()
        self.decoders: nn.Module = nn.Identity()
        self.temp_dec: nn.Module = nn.Identity()
        self.lookback = nn.Identity()

    # ---- lazy network build ----------------------------------------- #
    def setup(self, stage: str):
        if stage != 'fit' or not isinstance(self.encoders, nn.Identity):
            return  # already built or not training

        # grab one batch to infer dims
        sample = self.train_dataset[0]

        L = self.hparams.input_length
        H = len(self.hparams.h)
        Dy = 1  # target dim (univariate)
        if sample['past_covariates'] is not None:
            Dp = sample['past_covariates'].shape[-1]  # past covs  (might be 0)
        else:
            Dp = 0
        if sample['future_covariates'] is not None:
            Df = sample['future_covariates'].shape[-1]  # futr covs  (might be 0)
        else:
            Df = 0
        if sample['static_covariates'] is not None:
            Ds = sample['static_covariates'].shape[-1]
        else:
            Ds = 0

        # lengths
        enc_inp_dim = L * Dy + L * Dp + (L + H) * Df + (Ds * L) * self.hparams.use_static_covariates  # flattened input to encoder
        dec_out_dim = self.hparams.decoder_width  # TiDE's D'
        hidden = self.hparams.hidden_size

        # ---------- encoder ------------------------------------------ #
        enc_layers = [
            _Residual(enc_inp_dim, hidden, hidden, dropout=self.hparams.dropout, layer_norm=self.hparams.layer_norm)
        ]
        for _ in range(self.hparams.num_encoder_layers - 1):
            enc_layers.append(
                _Residual(hidden, hidden, hidden, dropout=self.hparams.dropout, layer_norm=self.hparams.layer_norm)
            )
        self.encoders = nn.Sequential(*enc_layers)

        # ---------- decoder ------------------------------------------ #
        dec_layers = []
        for _ in range(self.hparams.num_decoder_layers - 1):
            dec_layers.append(
                _Residual(hidden, hidden, hidden, dropout=self.hparams.dropout, layer_norm=self.hparams.layer_norm)
            )
        dec_layers.append(
            _Residual(hidden, dec_out_dim * H, hidden, dropout=self.hparams.dropout, layer_norm=self.hparams.layer_norm)
        )
        self.decoders = nn.Sequential(*dec_layers)

        # ---------- temporal decoder --------------------------------- #
        self.temp_dec = _Residual(
            inp=dec_out_dim + Df,  # concat last‑period future covs
            out=Dy,  # point forecast
            hidden=self.hparams.temporal_hidden,
            dropout=self.hparams.dropout,
            layer_norm=self.hparams.layer_norm,
        )

        # look‑back skip (linear across time)
        self.lookback = nn.Linear(L, H)

    # ---- forward expects your batch --------------------------------- #
    def forward(self, batch):
        """
        Returns tensor [B,H] with point forecasts.
        """
        y_hist: torch.Tensor = batch['past_target']  # [B,L,1]
        Xp = batch['past_covariates']  # [B,L,Dp] (maybe empty)
        Xfhist = batch['historic_future_covariates']  # [B,L,Df] (maybe empty)
        Xf = batch['future_covariates'] # [B,H,Df] (maybe empty)
        Xs = batch['static_covariates']

        parts = [y_hist]
        if Xp is not None:
            parts.append(Xp)
        if Xfhist is not None:
            parts.append(Xfhist)
        if self.hparams.use_static_covariates and Xs is not None:
            parts.append(Xs)
        enc_input = torch.cat(parts, dim=2)  # [B,L, Dy+Dp+Df]
        flat_hist = enc_input.flatten(start_dim=1)  # [B, L*(...)]
        parts = [flat_hist]
        if Xf is not None:
            flat_futr = Xf.flatten(start_dim=1)  # [B, H*Df]
            parts.append(flat_futr)
        enc = torch.cat(parts, dim=1)  # [B, enc_inp_dim]
        z = self.encoders(enc)  # [B, hidden]
        dec = self.decoders(z)  # [B, H*D'*1]
        dec = dec.view(z.size(0), len(self.hparams.h), -1)  # [B,H,D']
        last_futr = Xf  # [B,H,Df]
        if last_futr is not None and last_futr.size(-1):
            temp_in = torch.cat([dec, last_futr], dim=2)  # [B,H,D'+Df]
        else:
            temp_in = dec
        y = self.temp_dec(temp_in)  # [B,H,1]
        y = y.squeeze(-1)  # [B,H]
        # add look‑back skip
        skip = self.lookback(y_hist.transpose(1, 2).float()).transpose(1, 2)  # [B,H,1]→ squeeze
        skip = skip.squeeze(-1)
        return y + skip
