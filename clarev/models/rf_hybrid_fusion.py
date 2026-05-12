import torch
import torch.nn as nn


class RFFusionClassifier(nn.Module):
    """Fuse ClareV embedding features with a fixed RF probability feature.

    Two pathways combine additively at the logit level:
      1. A configurable embedding aggregator + concat-fusion MLP that mixes
         (compressed embedding, rf_prob).
      2. A deterministic RF-logit residual skip with a learnable scale.

    The skip guarantees that when the embedding branch carries little signal
    (small-N regimes such as Gastric), the classifier degrades toward the raw
    RF baseline rather than below it.

    Aggregation variants (controlled by ``agg_type``):

    - ``"flatten"`` (legacy default): treat the K x emb_dim matrix as a flat
      vector. Linear(K*emb_dim -> bottleneck_dim). Ignores V-gene structure
      and is heavily overparameterized on small N.
    - ``"per_v"``: shared Linear(emb_dim -> per_v_dim) applied to every V row,
      then flatten to K*per_v_dim and Linear -> bottleneck_dim. Respects V
      structure and reduces parameter count significantly.
    - ``"per_v_attn"``: shared Linear(emb_dim -> per_v_dim) applied per V,
      then attention pool over K -> per_v_dim, then Linear -> bottleneck_dim.
      Lets the supervised classifier learn which V genes matter for the task.
    """

    def __init__(
        self,
        v_num: int = 21,
        emb_dim: int = 120,
        class_num: int = 2,
        agg_type: str = "flatten",
        per_v_dim: int = 16,
        bottleneck_dim: int = 128,
        emb_dropout: float = 0.3,
        fusion_dropout: float = 0.2,
    ):
        super().__init__()
        self.class_num = class_num
        self.agg_type = agg_type
        self.per_v_dim = per_v_dim
        self.bottleneck_dim = bottleneck_dim

        if agg_type == "flatten":
            self.per_v_proj = None
            self.attn = None
            in_dim = v_num * emb_dim
        elif agg_type == "per_v":
            self.per_v_proj = nn.Linear(emb_dim, per_v_dim)
            self.attn = None
            in_dim = v_num * per_v_dim
        elif agg_type == "per_v_attn":
            self.per_v_proj = nn.Linear(emb_dim, per_v_dim)
            attn_hidden = max(per_v_dim // 2, 4)
            self.attn = nn.Sequential(
                nn.Linear(per_v_dim, attn_hidden),
                nn.Tanh(),
                nn.Linear(attn_hidden, 1),
            )
            in_dim = per_v_dim
        else:
            raise ValueError(
                f"Unknown agg_type={agg_type!r}. "
                "Choose from 'flatten', 'per_v', 'per_v_attn'."
            )

        self.bottleneck = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(emb_dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(bottleneck_dim + 1, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(bottleneck_dim, class_num),
        )

        # Learnable scale on the deterministic RF-logit residual. Init=1.0
        # so the model starts from "logits == RF baseline + zero-residual MLP".
        self.rf_skip_scale = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _prob_to_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = p.clamp(eps, 1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    def _aggregate_emb(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce (B, K, emb_dim) -> (B, in_dim) according to agg_type."""
        if self.agg_type == "flatten":
            return x.flatten(start_dim=1)

        # per_v / per_v_attn share the per-V projection
        h = self.per_v_proj(x)  # (B, K, per_v_dim)

        if self.agg_type == "per_v":
            return h.flatten(start_dim=1)  # (B, K*per_v_dim)

        # per_v_attn: attention pool over K
        attn_logits = self.attn(h).squeeze(-1)  # (B, K)
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, K)
        return (h * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, per_v_dim)

    def forward(self, x: torch.Tensor, rf_prob: torch.Tensor) -> torch.Tensor:
        if rf_prob.dim() == 1:
            rf_prob = rf_prob.unsqueeze(1)
        rf_prob = rf_prob.float()

        emb = self._aggregate_emb(x)
        main_feat = self.bottleneck(emb)
        combined = torch.cat([main_feat, rf_prob], dim=1)
        fusion_logit = self.fusion(combined)

        # Build a class_num-wide logit vector from the RF prob: only the
        # positive class (index 1) carries the RF signal; others remain 0.
        rf_logit = self._prob_to_logit(rf_prob)  # (B, 1)
        rf_logit_full = torch.zeros(
            rf_logit.size(0), self.class_num,
            device=rf_logit.device, dtype=rf_logit.dtype,
        )
        rf_logit_full[:, 1:2] = rf_logit

        return fusion_logit + self.rf_skip_scale * rf_logit_full
