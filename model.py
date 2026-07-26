"""
model.py
========

RIR-Former architecture.

IMPORTANT PROVENANCE NOTE: this implementation matches the architecture in
the OFFICIAL released evaluation script
(https://github.com/ShaoHenry/RIR-Former/blob/main/eval_rirformer.py), which
is the ground truth for reproducing the paper's Table 1/2 numbers -- NOT a
literal reading of Sec. 3 / Fig. 2's prose, which turned out to be
imprecise/simplified relative to what the authors actually built and
evaluated. Concretely, versus a naive reading of the paper text:

  * The Geometric Encoder's sinusoidal encoding DOES include the raw
    coordinate (`include_input=True`), even though Eq. 8's written formula
    only lists sin/cos terms.
  * The per-point input token is formed by ADDING the geometric and signal
    embeddings (`rir_tok + geo_tok`), not concatenating them, even though
    the text writes `o_m = [gamma(x_m); e_m]` (concatenation notation).
  * Each of the T temporal-segment "branches" is a FULLY INDEPENDENT
    pipeline -- its own geometric projection, its own signal encoder, and
    its own multi-layer Transformer encoder -- NOT a shared encoder with
    only the final decoder split into parallel heads (a literal reading of
    Fig. 2's caption suggests the latter; the released code confirms the
    former).
  * Self-attention uses a `key_padding_mask` so that missing/query points
    (whose RIR input is zeroed) are excluded as attention *keys* -- i.e.
    other tokens cannot attend to a query's meaningless zero-valued
    features -- while still fully participating as attention *queries*.
    This detail is not mentioned anywhere in the paper text at all.
  * There is NO residual/denoising refinement module in the released model,
    despite Sec. 3's prose mentioning a "lightweight residual denoising
    module". It's kept here as an optional, default-OFF component for
    experimentation, but is not part of the reproduced architecture.

Reconstruction is only required at the missing (query) positions; observed
positions are passed straight through in the final fused output (Eq. 10).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from config import Config, ModelConfig, get_config


# --------------------------------------------------------------------------- #
# Geometric encoding (Eq. 8) -- matches the released code: includes the raw
# coordinate alongside the sin/cos pairs (include_input=True by default).
# --------------------------------------------------------------------------- #
class SinusoidalPositionEncoding(nn.Module):
    """
    gamma(x) = [x, sin(2^0 pi x), cos(2^0 pi x), ..., sin(2^(i-1) pi x), cos(2^(i-1) pi x)]
    applied independently to each of the 3 coordinates, i = num_freqs (default 6).
    """

    def __init__(self, num_freqs: int = 6, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.register_buffer(
            "freq_bands", (2.0 ** torch.arange(num_freqs).float()) * math.pi
        )

    @property
    def out_dim_per_coord(self) -> int:
        return (1 + 2 * self.num_freqs) if self.include_input else 2 * self.num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 3) -> (..., 3 * out_dim_per_coord)
        parts = [x] if self.include_input else []
        for freq in self.freq_bands:
            parts.append(torch.sin(freq * x))
            parts.append(torch.cos(freq * x))
        return torch.cat(parts, dim=-1)


# --------------------------------------------------------------------------- #
# Transformer encoder block (self-attention + FF, Vaswani et al. [26])
# --------------------------------------------------------------------------- #
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D) tokens for the whole array (observed + query points).
        obs_mask: (B, L), 1 = observed, 0 = missing/query.

        `key_padding_mask=obs_mask==0` excludes missing/query points as
        attention KEYS (their zeroed-RIR-derived features carry no real
        signal and would otherwise dilute what every other token -- including
        other queries -- attends to). Missing/query points still fully
        participate as QUERIES, so they still receive a contextual output
        built entirely from the real, observed microphones.
        """
        key_padding_mask = obs_mask == 0
        x2, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.drop(x2))
        x2 = self.ff(x)
        return self.norm2(x + self.drop(x2))

# --------------------------------------------------------------------------- #
# OPTIONAL EXTENSION (Strategy 2, default OFF): geometry injected as an
# additive attention bias, computed once per branch and reused across every
# transformer layer. This is purely additive to the existing architecture --
# rir_tok/geo_tok token construction is untouched; this only changes how
# attention logits are computed, and only when explicitly enabled.
# --------------------------------------------------------------------------- #
class GeoAttentionBias(nn.Module):
    """Computes a (B, n_heads, L, L) additive bias for attention logits from
    pairwise relative geometry. Shared across all layers within a branch."""

    def __init__(self, n_heads: int = 4, d_edge_hidden: int = 64,
                 speed_of_sound: float = 343.0):
        super().__init__()
        self.speed_of_sound = speed_of_sound
        self.n_heads = n_heads
        self.edge_mlp = nn.Sequential(
            nn.Linear(5, d_edge_hidden),
            nn.GELU(),
            nn.Linear(d_edge_hidden, n_heads),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B, L, 3) -> bias: (B, n_heads, L, L)
        rel_pos = coords[:, :, None, :] - coords[:, None, :, :]  # (B, L, L, 3)
        dist = rel_pos.norm(dim=-1, keepdim=True)                 # (B, L, L, 1)
        tof = dist / self.speed_of_sound                           # (B, L, L, 1)
        edge_feat = torch.cat([rel_pos, dist, tof], dim=-1)        # (B, L, L, 5)
        bias = self.edge_mlp(edge_feat)                             # (B, L, L, n_heads)
        return bias.permute(0, 3, 1, 2)                              # (B, n_heads, L, L)


class TransformerEncoderBlockWithGeoBias(nn.Module):
    """Same role as TransformerEncoderBlock, but with a manual attention
    implementation so a geo_bias tensor can be added to the logits before
    the existing key_padding_mask is applied. Only instantiated when
    cfg.model.use_geo_attn_bias=True."""

    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, obs_mask: torch.Tensor,
                geo_bias: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, n_heads, L, d_head)

        logits = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)   # (B, n_heads, L, L)
        logits = logits + geo_bias                                    # geometry injected here

        # same masking semantics as TransformerEncoderBlock: missing/query
        # points excluded as KEYS, still fully valid as QUERIES.
        key_mask = (obs_mask == 0)[:, None, None, :]
        logits = logits.masked_fill(key_mask, float("-inf"))

        attn = logits.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)

        x = self.norm1(x + self.drop(out))
        x2 = self.ff(x)
        return self.norm2(x + self.drop(x2))

# --------------------------------------------------------------------------- #
# Per-segment decoder branch -- a FULLY INDEPENDENT encoder+transformer+decoder
# pipeline per temporal segment, matching the released reference code.
# --------------------------------------------------------------------------- #
class RIRBranch(nn.Module):
    def __init__(self, K: int, segment_len: int, d_model: int = 256,
                 n_layers: int = 3, n_heads: int = 4, pos_freqs: int = 6,
                 dropout: float = 0.1, use_geo_attn_bias: bool = False,
                 speed_of_sound: float = 343.0):
        super().__init__()

        self.pos_enc = SinusoidalPositionEncoding(num_freqs=pos_freqs, include_input=True)
        coord_dim = 3
        pe_dim = coord_dim * self.pos_enc.out_dim_per_coord

        self.geo_proj = nn.Sequential(
            nn.Linear(pe_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.rir_encoder = nn.Sequential(
            nn.Linear(K, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.use_geo_attn_bias = use_geo_attn_bias
        if use_geo_attn_bias:
            self.geo_attn_bias = GeoAttentionBias(
                n_heads=n_heads, speed_of_sound=speed_of_sound)
            self.blocks = nn.ModuleList([
                TransformerEncoderBlockWithGeoBias(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                TransformerEncoderBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ])

        self.decoder = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, segment_len),
        )

    def forward(self, H_masked: torch.Tensor, mask: torch.Tensor,
                geo_feat: torch.Tensor) -> torch.Tensor:
        """
        H_masked: (B, L, K) -- observed RIRs with missing rows zeroed.
        mask:     (B, L)    -- 1 = observed, 0 = missing/query.
        geo_feat: (B, L, 3) -- per-sample-shifted microphone/target coordinates.
        returns:  (B, L, segment_len)
        """
        B, L, K = H_masked.shape

        rir_tok = self.rir_encoder(H_masked.reshape(B * L, K)).view(B, L, -1)
        geo_tok = self.geo_proj(self.pos_enc(geo_feat))

        h = rir_tok + geo_tok  # ADDITION, matching the released code exactly
        if self.use_geo_attn_bias:
            geo_bias = self.geo_attn_bias(geo_feat)   # (B, n_heads, L, L), computed once
            for blk in self.blocks:
                h = blk(h, mask, geo_bias)             # reused at every layer
        else:
            for blk in self.blocks:
                h = blk(h, mask)

        return self.decoder(h)


# --------------------------------------------------------------------------- #
# Optional lightweight residual refinement module -- NOT part of the released
# model (see module docstring); default OFF so that build_model(cfg) with
# defaults reproduces the official architecture exactly. Kept for anyone who
# wants to experiment with the "lightweight residual denoising module"
# mentioned in the paper's prose but absent from the released code.
# --------------------------------------------------------------------------- #
class ResidualRefineModule(nn.Module):
    def __init__(self, channels: int = 16, kernel_size: int = 9):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size, padding=pad),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.GELU(),
            nn.Conv1d(channels, 1, kernel_size, padding=pad),
        )

    def forward(self, H_hat_raw: torch.Tensor) -> torch.Tensor:
        B, L, K = H_hat_raw.shape
        x = H_hat_raw.reshape(B * L, 1, K)
        residual = self.net(x).reshape(B, L, K)
        return H_hat_raw + residual


# --------------------------------------------------------------------------- #
# Full RIR-Former model
# --------------------------------------------------------------------------- #
class RIRFormer(nn.Module):
    def __init__(self, K: int, d_model: int = 256, n_layers: int = 3,
                 n_heads: int = 4, n_segments: int = 4, pos_freqs: int = 6,
                 dropout: float = 0.1, use_residual_refine: bool = False,
                 use_geo_attn_bias: bool = False,
                 speed_of_sound: float = 343.0):
        super().__init__()
        assert K % n_segments == 0, "K must be divisible by n_segments"

        self.K = K
        self.n_segments = n_segments
        self.segment_len = K // n_segments

        self.branches = nn.ModuleList([
            RIRBranch(K=K, segment_len=self.segment_len, d_model=d_model,
                      n_layers=n_layers, n_heads=n_heads, pos_freqs=pos_freqs,
                      dropout=dropout,
                      use_geo_attn_bias=use_geo_attn_bias,
                      speed_of_sound=speed_of_sound)
            for _ in range(n_segments)
        ])

        # Default False: matches the released reference model exactly.
        self.use_residual_refine = use_residual_refine
        if use_residual_refine:
            self.refine = ResidualRefineModule()

    def forward(self, H_norm: torch.Tensor, mask: torch.Tensor,
                geo_feat: torch.Tensor) -> torch.Tensor:
        """
        H_norm:   (B, L, K) normalized RIRs (ground truth at every position;
                  only the `mask==1` rows are actually fed in as input).
        mask:     (B, L) 1 = observed mic, 0 = missing/query target.
        geo_feat: (B, L, 3) per-sample-shifted coordinates for every point.
        returns:  (B, L, K) reconstructed RIRs (observed rows pass through
                  unchanged, missing rows are the model's predictions).
        """
        H_masked = H_norm * mask.unsqueeze(-1)

        segments = [branch(H_masked, mask, geo_feat) for branch in self.branches]
        H_hat_raw = torch.cat(segments, dim=-1)

        if self.use_residual_refine:
            H_hat_raw = self.refine(H_hat_raw)

        m = mask.unsqueeze(-1)
        H_fused = H_norm * m + H_hat_raw * (1 - m)
        return H_fused

    def forward_segment_only(self, seg_idx: int, H_norm: torch.Tensor,
                              mask: torch.Tensor, geo_feat: torch.Tensor) -> torch.Tensor:
        """Run only branch `seg_idx` -- used by the per-segment decoder
        finetuning stage described in Sec. 3 ("Training Objective"). Safe
        to use for the cheap/isolated forward when use_residual_refine is
        False (the default / released-model setting), since concatenation
        alone does not mix information across branches; train.py switches
        to a full-forward-based loss automatically when refine is enabled,
        since the refine module DOES mix information across segment
        boundaries and would otherwise go stale relative to a branch
        finetuned in isolation."""
        H_masked = H_norm * mask.unsqueeze(-1)
        return self.branches[seg_idx](H_masked, mask, geo_feat)


def build_model(cfg: Config) -> RIRFormer:
    m = cfg.model
    return RIRFormer(
        K=cfg.K,
        d_model=m.d_model,
        n_layers=m.n_layers,
        n_heads=m.n_heads,
        n_segments=m.n_segments,
        pos_freqs=m.pos_freqs,
        dropout=m.dropout,
        use_residual_refine=m.use_residual_refine,
        use_geo_attn_bias=m.use_geo_attn_bias,
        speed_of_sound=cfg.room.speed_of_sound,
    )


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    print("Running model.py self-test...")

    # --- SinusoidalPositionEncoding (matches released code: includes raw x) ---
    pe = SinusoidalPositionEncoding(num_freqs=6)
    x = torch.randn(2, 5, 3)
    y = pe(x)
    assert y.shape == (2, 5, 3 * (1 + 2 * 6)), y.shape
    print(f"[OK] SinusoidalPositionEncoding (include_input=True, matches "
          f"released code): {tuple(x.shape)} -> {tuple(y.shape)}")

    # --- TransformerEncoderBlock: key_padding_mask actually excludes missing keys ---
    torch.manual_seed(1)
    blk = TransformerEncoderBlock(d_model=16, n_heads=2)
    blk.eval()
    B, L, D = 1, 6, 16
    h = torch.randn(B, L, D)
    mask_a = torch.tensor([[1., 1., 1., 0., 0., 0.]])  # last 3 are missing/query
    out_a = blk(h, mask_a)

    # Perturbing only the MISSING tokens' input must not change the OBSERVED
    # tokens' output at all (since missing points are excluded as keys) --
    # this is the key behavioral difference vs. plain unmasked self-attention.
    h_perturbed = h.clone()
    h_perturbed[:, 3:] += torch.randn(B, 3, D) * 10.0
    out_b = blk(h_perturbed, mask_a)
    assert torch.allclose(out_a[:, :3], out_b[:, :3], atol=1e-5), \
        "Observed tokens' output should be invariant to changes in missing/query tokens"
    assert not torch.allclose(out_a[:, 3:], out_b[:, 3:], atol=1e-5), \
        "Missing/query tokens' own output SHOULD change (they still attend as queries)"
    print("[OK] key_padding_mask correctly excludes missing/query points as attention "
          "keys (observed-token outputs are invariant to changes in missing tokens)")

    # --- RIRBranch: additive token combination ---
    K, seg_len = 64, 16
    branch = RIRBranch(K=K, segment_len=seg_len, d_model=32, n_layers=2, n_heads=4)
    Bt, Lt = 3, 12
    H_masked = torch.randn(Bt, Lt, K)
    mask = torch.ones(Bt, Lt)
    geo = torch.randn(Bt, Lt, 3)
    seg_out = branch(H_masked, mask, geo)
    assert seg_out.shape == (Bt, Lt, seg_len), seg_out.shape
    print(f"[OK] RIRBranch output shape: {tuple(seg_out.shape)}")

    # --- Full RIRFormer, both experiment configs, matching released defaults ---
    for exp in ("exp1", "exp2"):
        cfg = get_config(exp, **{"model.d_model": 32, "model.n_layers": 2,
                                  "model.n_heads": 4, "model.n_segments": 4})
        assert cfg.model.use_residual_refine is False, \
            "Default config should match the released model (no refine module)"
        model = build_model(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        assert not hasattr(model, "refine"), \
            "refine module should not exist when use_residual_refine=False"

        Bt, Lt = 2, cfg.array.n_points
        H_norm = torch.randn(Bt, Lt, cfg.K)
        geo_feat = torch.randn(Bt, Lt, 3)
        mask = torch.ones(Bt, Lt)
        mask[:, : int(0.7 * Lt)] = 0.0  # 70% missing, matches paper's default MR

        out = model(H_norm, mask, geo_feat)
        assert out.shape == (Bt, Lt, cfg.K), out.shape
        assert torch.isfinite(out).all(), "Model output contains non-finite values"

        obs_idx = (mask[0] == 1).nonzero(as_tuple=True)[0]
        assert torch.allclose(out[0, obs_idx], H_norm[0, obs_idx], atol=1e-6), \
            "Observed positions should be passed through unchanged"

        loss = ((out - H_norm) ** 2 * (1 - mask).unsqueeze(-1)).mean()
        loss.backward()
        grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        assert grad_norm > 0, "No gradient flowed through the model"

        print(f"[OK] RIRFormer [{exp}]: K={cfg.K}, params={n_params:,}, "
              f"out={tuple(out.shape)}, loss={loss.item():.4f}, grad_norm={grad_norm:.2f}, "
              f"use_residual_refine={model.use_residual_refine}")

        seg_out = model.forward_segment_only(0, H_norm, mask, geo_feat)
        assert seg_out.shape == (Bt, Lt, model.segment_len), seg_out.shape
        print(f"[OK] forward_segment_only(0): {tuple(seg_out.shape)}")

    # --- Optional refine module still works when explicitly enabled ---
    cfg_refine = get_config("exp1", **{"model.d_model": 32, "model.n_layers": 1,
                                        "model.n_heads": 2, "model.use_residual_refine": True})
    model_refine = build_model(cfg_refine)
    assert hasattr(model_refine, "refine")
    Lt = cfg_refine.array.n_points
    out_r = model_refine(torch.randn(2, Lt, cfg_refine.K), torch.ones(2, Lt),
                          torch.randn(2, Lt, 3))
    assert out_r.shape == (2, Lt, cfg_refine.K)
    print("[OK] Optional residual refine module still available via "
          "model.use_residual_refine=True for experimentation")

    print("\nAll model.py self-tests passed!")