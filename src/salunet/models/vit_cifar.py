"""Configurable copy of the user's working ViT-CIFAR architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedSALUGELU(nn.Module):
    """Apply fixed SALU to GELU(x), without adding trainable parameters."""

    def __init__(self, a: float = 1.0, b: float = 0.001) -> None:
        super().__init__()
        if a <= 0 or b <= 0:
            raise ValueError("FixedSALUGELU requires strictly positive a and b.")
        self.register_buffer("a", torch.tensor(float(a), dtype=torch.float32))
        self.register_buffer("b", torch.tensor(float(b), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gelu = F.gelu(x)
        return self.a * gelu * torch.rsqrt(1.0 + self.a * self.b * gelu.square())


def _normalization(name: str, hidden: int, dyt_alpha_init: float = 0.5,
                   salu_kwargs=None) -> nn.Module:
    from salunet.layers.vit_salu import ViTSALU
    from salunet.signal_control.dyt import DyT
    if name == "ln":
        return nn.LayerNorm(hidden)
    if name == "rmsnorm":
        # Explicit epsilon matches the supplied implementation.
        return nn.RMSNorm(hidden, eps=1e-6)
    if name == "salu":
        return ViTSALU(**(salu_kwargs or {}))
    if name == "dyt":
        return DyT(dim=hidden, alpha_init=dyt_alpha_init, affine=True)
    raise ValueError("ViT normalization must be ln, rmsnorm, salu, or dyt.")


def _activation(name: str, gate_positivity: str = "abs_eps_legacy",
                gate_a_init: float = 1.0, gate_b_init: float = 1.0) -> nn.Module:
    from salunet.layers.vit_salu import ViTSALU
    from salunet.activations.swalu import SWALU
    from salunet.activations.galu import GALU
    if name == "gelu":
        return nn.GELU()
    if name == "salu_gelu_fixed":
        return FixedSALUGELU(a=gate_a_init, b=gate_b_init)
    if name not in {"swalu", "galu"}:
        raise ValueError("ViT activation must be gelu, salu_gelu_fixed, swalu, or galu.")
    # 'abs' in the supplied ViT means abs(raw)+eps, not the CNN clamp mode.
    positivity = "abs_eps_legacy" if gate_positivity == "abs" else gate_positivity
    gate = (SWALU if name == "swalu" else GALU)(
        a_init=gate_a_init, b_init=gate_b_init, positivity=positivity)
    gate.salu = ViTSALU(a_init=gate_a_init, b_init=gate_b_init,
                       positivity=positivity)
    return gate


class MultiHeadSelfAttention(nn.Module):
    """Multi-head attention scaled by the square root of head dimension."""

    def __init__(self, feats: int, head: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        if feats % head:
            raise ValueError(f"feats={feats} must be divisible by head={head}.")
        self.head = head
        self.feats = feats
        self.sqrt_d = (self.feats // self.head)**0.5
        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)
        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.size()
        q = self.q(x).view(batch, tokens, self.head, self.feats // self.head).transpose(1, 2)
        k = self.k(x).view(batch, tokens, self.head, self.feats // self.head).transpose(1, 2)
        v = self.v(x).view(batch, tokens, self.head, self.feats // self.head).transpose(1, 2)
        score = F.softmax(
            torch.einsum("bhif, bhjf->bhij", q, k) / self.sqrt_d, dim=-1
        )
        attention = torch.einsum("bhij, bhjf->bihf", score, v)
        return self.dropout(self.o(attention.flatten(2)))


class TransformerEncoder(nn.Module):
    """Pre-norm encoder with one activation between the MLP projections."""

    def __init__(
        self,
        feats: int,
        mlp_hidden: int,
        head: int = 8,
        dropout: float = 0.0,
        normalization: str = "salu",
        activation: str = "swalu",
        dyt_alpha_init: float = 0.5,
        gate_positivity: str = "abs",
        gate_a_init: float = 1.0,
        gate_b_init: float = 1.0,
        salu_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.la1 = _normalization(normalization, feats, dyt_alpha_init, salu_kwargs)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = _normalization(normalization, feats, dyt_alpha_init, salu_kwargs)
        first_activation = _activation(activation, gate_positivity,
                                       gate_a_init, gate_b_init)
        mlp_layers = [
            nn.Linear(feats, mlp_hidden),
            first_activation,
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
        ]
        mlp_layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.msa(self.la1(x)) + x
        return self.mlp(self.la2(out)) + out


class SaluNetTCIFAR(nn.Module):
    """CIFAR ViT with configurable stabilizers and activation gates."""

    def __init__(
        self,
        num_classes: int,
        normalization: str = "salu",
        activation: str = "swalu",
        image_size: int = 32,
        patches_per_axis: int = 8,
        hidden: int = 384,
        mlp_hidden: int = 384,
        heads: int = 12,
        layers: int = 7,
        dropout: float = 0.0,
        class_token: bool = True,
        dyt_alpha_init: float = 0.5,
        gate_positivity: str = "abs",
        gate_a_init: float = 1.0,
        gate_b_init: float = 1.0,
        salu_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        if image_size <= 0 or patches_per_axis <= 0 or image_size % patches_per_axis:
            raise ValueError("image_size must be divisible by patches_per_axis.")
        self.image_size = image_size
        self.patch = patches_per_axis
        self.is_cls_token = class_token
        self.patch_size = image_size // self.patch
        patch_features = self.patch_size**2 * 3
        num_tokens = self.patch**2 + 1 if self.is_cls_token else self.patch**2

        self.emb = nn.Linear(patch_features, hidden)
        self.cls_token = (
            nn.Parameter(torch.randn(1, 1, hidden)) if self.is_cls_token else None
        )
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden))
        self.enc = nn.Sequential(
            *[
                TransformerEncoder(
                    hidden,
                    mlp_hidden=mlp_hidden,
                    dropout=dropout,
                    head=heads,
                    normalization=normalization,
                    activation=activation,
                    dyt_alpha_init=dyt_alpha_init,
                    gate_positivity=gate_positivity,
                    gate_a_init=gate_a_init,
                    gate_b_init=gate_b_init,
                    salu_kwargs=salu_kwargs,
                )
                for _ in range(layers)
            ]
        )
        self.fc = nn.Sequential(
            _normalization(normalization, hidden, dyt_alpha_init, salu_kwargs),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError("Expected NCHW RGB input matching image_size.")
        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        out = out[:, 0] if self.is_cls_token else out.mean(1)
        return self.fc(out)

    def _to_words(self, x: torch.Tensor) -> torch.Tensor:
        out = (
            x.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 4, 5, 1)
        )
        return out.reshape(x.size(0), self.patch**2, -1)


from salunet.models.registry import register_model

@register_model("vit_cifar")
def build_salu_net_t_cifar(num_classes, signal_control="salu", activation="galu",
                          salu_kwargs=None, activation_kwargs=None,
                          bn_index_overrides=None, compile_model=False, compile_mode="default",
                          **kwargs):
    """Build the same backbone for every normalization/activation comparison.

    Compilation is handled by the generic Lightning module after device setup.
    """
    if bn_index_overrides:
        raise ValueError("BN index overrides do not apply to vit_cifar.")
    gate = activation_kwargs or {}
    return SaluNetTCIFAR(
        num_classes=num_classes, normalization=signal_control, activation=activation,
        salu_kwargs=salu_kwargs,
        gate_positivity=gate.get("positivity", "abs_eps_legacy"),
        gate_a_init=gate.get("a_init", 1.0), gate_b_init=gate.get("b_init", 1.0),
        **kwargs,
    )
