"""
DEM-MoE: Demographic-Aware Mixture of Experts for annotation disagreement.

Adapted from Xu et al. (ACL 2026) https://github.com/Nancy078/DEM-MoE
for classification (No/Unsure/Yes or binary) on pre-extracted text embeddings.

Key differences from the original:
  - Input: pre-extracted sentence embeddings (Xi .npy) instead of ModernBERT CLS
  - Output: K-class softmax distribution (not regression scalar)
  - Loss: KL divergence vs soft Yi label distribution (not MSE)
  - Identity features: derived from argmax of one-hot A-matrix blocks
  - Framework: PyTorch (standalone, not TF like DiADEM/DisCo)

Architecture (same topology as original):
  x = [text_emb ; ann_emb ; id_emb_projected]
  gating(x) → Gumbel-softmax top-K → weighted sum of K expert MLPs
  → classification head (softmax over num_classes)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianAnnotatorEmbedding(nn.Module):
    """
    Diagonal-Gaussian variational embedding for a single categorical variable.
    Each category: learnable mu + log_var; sample via reparameterisation trick.
    KL against N(0,I) returned for regularisation.
    """
    def __init__(self, num_categories: int, embedding_dim: int):
        super().__init__()
        self.mu      = nn.Embedding(num_categories, embedding_dim)
        self.log_var = nn.Embedding(num_categories, embedding_dim)
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.constant_(self.log_var.weight, -2.0)  # small initial variance

    def forward(self, ids: torch.Tensor):
        """
        ids: (B,) LongTensor of category indices
        Returns: z (B, dim), kl_loss scalar
        """
        mu      = self.mu(ids)
        log_var = self.log_var(ids)
        std     = torch.exp(0.5 * log_var)
        z       = mu + torch.randn_like(std) * std
        kl      = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        return z, kl


class BayesianIdentityEmbedding(nn.Module):
    """
    One BayesianAnnotatorEmbedding per categorical demographic attribute;
    outputs are concatenated into a single identity vector.
    """
    def __init__(self, num_categories_list: list, embedding_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([
            BayesianAnnotatorEmbedding(n, embedding_dim)
            for n in num_categories_list
        ])

    def forward(self, identity_ids: torch.Tensor):
        """
        identity_ids: (B, num_attributes) LongTensor
        Returns: concat embedding (B, dim * num_attributes), sum of KLs
        """
        zs, kls = [], []
        for i, layer in enumerate(self.layers):
            z, kl = layer(identity_ids[:, i])
            zs.append(z)
            kls.append(kl)
        return torch.cat(zs, dim=1), torch.stack(kls).sum()


class DemMoEClassifier(nn.Module):
    """
    DEM-MoE adapted for classification over pre-extracted text embeddings.

    Input dimensions
    ----------------
    text_embed_dim  : dimensionality of Xi embeddings (e.g. 768)
    num_annotators  : unique annotator count (for Bayesian annotator emb)
    num_cat_per_attr: list[int] — vocabulary size for each CATEGORICAL
                      demographic attribute (derived from metadata_encoding_config)
    num_cont_feats  : number of continuous demographic features (e.g. moral values)
    num_classes     : output classes (2 for binary, 3 for No/Unsure/Yes)

    Routing
    -------
    Gumbel-softmax with temperature annealing (same schedule as original):
      T(epoch) = max(T_min, T_init * T_decay^(epoch - T_warmup))
    Top-K experts selected per sample, load-balanced with per-expert biases.
    """

    def __init__(
        self,
        text_embed_dim:    int,
        num_annotators:    int,
        num_cat_per_attr:  list,        # e.g. [3, 21, 3, 8, 10] for d3code
        num_cont_feats:    int = 0,     # continuous demographic features
        num_classes:       int = 3,
        hidden_dim:        int = 256,
        num_experts:       int = 8,
        ann_embed_dim:     int = 128,
        id_embed_dim:      int = 64,    # per attribute
        projected_id_dim:  int = 256,
        dropout:           float = 0.3,
        top_k:             int = 3,
        bias_lr:           float = 0.05,
        T_init:            float = 6.0,
        T_decay:           float = 0.92,
        T_min:             float = 4.0,
        T_warmup:          int = 25,
        overflow_on:       bool = True,
        capacity_scale:    float = 1.0,
    ):
        super().__init__()

        self.num_experts    = num_experts
        self.top_k          = top_k
        self.hidden_dim     = hidden_dim
        self.bias_lr        = bias_lr
        self.T_init         = T_init
        self.T_decay        = T_decay
        self.T_min          = T_min
        self.T_warmup       = T_warmup
        self.overflow_on    = overflow_on
        self.capacity_scale = capacity_scale
        self.num_classes    = num_classes

        self.register_buffer("expert_biases", torch.zeros(num_experts, dtype=torch.float32))

        # ── Embeddings ─────────────────────────────────────────────────────────
        self.annotator_emb = BayesianAnnotatorEmbedding(num_annotators, ann_embed_dim)
        self.identity_emb  = BayesianIdentityEmbedding(num_cat_per_attr, id_embed_dim)

        identity_raw_dim = id_embed_dim * len(num_cat_per_attr)
        self.id_proj      = nn.Linear(identity_raw_dim, projected_id_dim)

        # Continuous feature projection (moral values etc.)
        self.has_cont = num_cont_feats > 0
        if self.has_cont:
            self.cont_proj = nn.Linear(num_cont_feats, 64)
            cont_out_dim   = 64
        else:
            cont_out_dim   = 0

        # ── MoE input dim = text + ann + id_proj + cont ────────────────────────
        moe_input_dim = text_embed_dim + ann_embed_dim + projected_id_dim + cont_out_dim

        # ── Layer norms ────────────────────────────────────────────────────────
        self.text_ln = nn.LayerNorm(text_embed_dim)
        self.ann_ln  = nn.LayerNorm(ann_embed_dim)
        self.id_ln   = nn.LayerNorm(projected_id_dim)

        # ── Expert pool ────────────────────────────────────────────────────────
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(moe_input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),    nn.ReLU(),
            ) for _ in range(num_experts)
        ])

        # ── Gating ─────────────────────────────────────────────────────────────
        self.gate = nn.Linear(moe_input_dim, num_experts)
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.zeros_(self.gate.bias)

        # ── Output ─────────────────────────────────────────────────────────────
        self.output_dropout = nn.Dropout(dropout)
        self.classifier     = nn.Linear(hidden_dim, num_classes)

    # ──────────────────────────────────────────────────────────────────────────

    def _build_input(
        self,
        text_embeds:  torch.Tensor,   # (B, text_embed_dim)
        annotator_ids: torch.Tensor,  # (B,) long
        identity_ids:  torch.Tensor,  # (B, num_cat_attrs) long
        cont_feats:    torch.Tensor | None,  # (B, num_cont_feats) float
    ):
        text = self.text_ln(text_embeds)
        ann, kl_ann = self.annotator_emb(annotator_ids)
        ann  = self.ann_ln(ann)
        ide, kl_id  = self.identity_emb(identity_ids)
        ide  = self.id_ln(self.id_proj(ide))

        parts = [text, ann, ide]
        if self.has_cont and cont_feats is not None:
            parts.append(F.relu(self.cont_proj(cont_feats)))

        return torch.cat(parts, dim=1), kl_ann, kl_id

    def forward(
        self,
        text_embeds:   torch.Tensor,
        annotator_ids: torch.Tensor,
        identity_ids:  torch.Tensor,
        cont_feats:    torch.Tensor | None = None,
        epoch:         int | None = None,
        return_routing: bool = False,
    ):
        """
        Returns (compact):
            logits (B, num_classes), kl_ann, kl_id

        Returns (return_routing=True):
            logits, kl_ann, kl_id, topk_indices, topk_scores
        """
        x, kl_ann, kl_id = self._build_input(
            text_embeds, annotator_ids, identity_ids, cont_feats
        )

        # ── Gating with Gumbel annealing ───────────────────────────────────────
        expert_scores = self.gate(x) / 4.0 + self.expert_biases.unsqueeze(0)

        if self.training and epoch is not None:
            gumbel = -torch.empty_like(expert_scores).exponential_().log()
            if epoch < self.T_warmup:
                T = self.T_init
            else:
                T = max(self.T_min,
                        self.T_init * (self.T_decay ** (epoch - self.T_warmup)))
        else:
            gumbel, T = 0.0, 1.0

        softmax_scores           = torch.softmax((expert_scores + gumbel) / T, dim=-1)
        topk_scores, topk_indices = torch.topk(softmax_scores, self.top_k, dim=-1)

        # ── Capacity overflow ──────────────────────────────────────────────────
        B = x.shape[0]
        if self.overflow_on:
            primary_idx = topk_indices[:, 0]
            primary_w   = topk_scores[:, 0].clone()
            capacity    = math.ceil((B / self.num_experts) * self.capacity_scale)
            counts      = torch.bincount(primary_idx, minlength=self.num_experts)
            overflow    = torch.zeros(B, dtype=torch.bool, device=x.device)
            for e in range(self.num_experts):
                extra = int(counts[e].item()) - capacity
                if extra > 0:
                    pos      = torch.nonzero(primary_idx == e, as_tuple=False).flatten()
                    smallest = torch.topk(primary_w[pos], k=extra, largest=False).indices
                    overflow[pos[smallest]] = True
            primary_w[overflow] = 0.0
            topk_scores = topk_scores.clone()
            topk_scores[:, 0] = primary_w
            topk_scores = topk_scores / (topk_scores.sum(dim=1, keepdim=True) + 1e-9)

        # ── Expert computation ─────────────────────────────────────────────────
        outputs = torch.zeros(B, self.hidden_dim, device=x.device)
        for i in range(B):
            for k in range(self.top_k):
                eidx = topk_indices[i, k].item()
                gw   = topk_scores[i, k]
                h    = self.experts[eidx](x[i])
                outputs[i] += gw.unsqueeze(-1) * h

        logits = self.classifier(self.output_dropout(outputs))  # (B, num_classes)

        if return_routing:
            return logits, kl_ann, kl_id, topk_indices, topk_scores
        return logits, kl_ann, kl_id

    def predict_proba(
        self,
        text_embeds:   torch.Tensor,
        annotator_ids: torch.Tensor,
        identity_ids:  torch.Tensor,
        cont_feats:    torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inference: return softmax probabilities (B, num_classes)."""
        self.eval()
        with torch.no_grad():
            logits, _, _ = self.forward(
                text_embeds, annotator_ids, identity_ids, cont_feats, epoch=None
            )
            return torch.softmax(logits, dim=-1)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase-aware loss computation (mirrors Xu et al. 3-phase schedule)
    # ──────────────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        logits:        torch.Tensor,          # (B, num_classes)
        yi_targets:    torch.Tensor,          # (B, num_classes) soft label distribution
        kl_ann:        torch.Tensor,
        kl_id:         torch.Tensor,
        topk_indices:  torch.Tensor | None = None,
        topk_scores:   torch.Tensor | None = None,
        phase:         int = 1,
        kl_weight:     float = 1e-4,
        ortho_weight:  float = 0.01,
        var_weight:    float = 0.01,
        load_weight:   float = 0.01,
        class_weights: torch.Tensor | None = None,   # (num_classes,) inverse-freq weights
    ) -> tuple:
        """
        Three-phase loss matching the original paper schedule.

        Phase 1 (A): main KL + annotator KL + load balance
        Phase 2 (B): + orthogonality + variance losses (expert specialisation)
        Phase 3 (C): main KL + annotator KL only (frozen gating)

        Main loss: KL divergence of softmax(logits) vs Yi soft labels.
        This is better than cross-entropy for perspectivist models because
        it trains against the full annotation distribution, not just majority.

        Returns: (total_loss, {component_name: value})
        """
        # Main: KL(Yi || predicted) — train model to match annotator distribution
        log_probs = F.log_softmax(logits, dim=-1)
        yi_safe   = (yi_targets + 1e-8)
        yi_safe   = yi_safe / yi_safe.sum(dim=1, keepdim=True)
        if class_weights is not None:
            # Weight each class's contribution to the KL by inverse frequency
            kl_per_class = yi_safe * (torch.log(yi_safe) - log_probs)  # (B, K)
            main_loss    = (kl_per_class * class_weights).sum(dim=1).mean()
        else:
            main_loss = F.kl_div(log_probs, yi_safe, reduction="batchmean")

        # Bayesian KL regularisation
        kl_loss = kl_weight * (kl_ann + kl_id)

        components = {"main": main_loss.item(), "kl": kl_loss.item()}

        if phase in (1, 2) and topk_scores is not None:
            # Load-balance loss: encourage uniform expert usage
            avg_load  = topk_scores.mean(dim=0)
            load_loss = load_weight * torch.sum((avg_load - 1.0 / self.num_experts) ** 2)
            components["load"] = load_loss.item()
        else:
            load_loss = torch.zeros(1, device=logits.device)[0]

        if phase == 2 and topk_indices is not None:
            # Orthogonality loss: expert weight matrices should be orthogonal
            ortho_loss = torch.zeros(1, device=logits.device)[0]
            for exp in self.experts:
                W = exp[0].weight  # (hidden_dim, input_dim)
                G = W @ W.T
                I = torch.eye(G.shape[0], device=G.device)
                ortho_loss = ortho_loss + torch.norm(G - I, p="fro")
            ortho_loss = ortho_weight * ortho_loss / self.num_experts

            # Variance loss: encourage diverse routing distributions
            # High variance = experts specialise on different inputs
            routing_var = topk_scores.var(dim=0).mean()
            var_loss    = -var_weight * routing_var  # negative because we want HIGH variance

            components["ortho"] = ortho_loss.item()
            components["var"]   = var_loss.item()
        else:
            ortho_loss = torch.zeros(1, device=logits.device)[0]
            var_loss   = torch.zeros(1, device=logits.device)[0]

        total = main_loss + kl_loss + load_loss + ortho_loss + var_loss
        components["total"] = total.item()
        return total, components
