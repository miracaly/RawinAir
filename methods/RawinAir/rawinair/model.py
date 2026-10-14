from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import GraphData, sparse_mm


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class MLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner = dim * hidden_mult
        self.norm = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, inner)
        self.fc2 = nn.Linear(inner, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = F.silu(self.fc1(y))
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return x + y


@dataclass
class RawinAirOutput:
    y_pred: torch.Tensor  # (B, T_pred, N, C_out)
    transport_pred: torch.Tensor  # (B, T_pred, N, C_out) extracted from STD for DIC


class RawinAir(nn.Module):
    """
    RawinAir-style surrogate model for station-level AQF.

    Differences vs. paper (pragmatic):
    - STD uses normalized graph Laplacian diffusion on hidden states (sparse_mm).
    - DIC is applied on a learned "transport contribution" readout per node, enforcing:
        * spatial net-transport ~ 0 (sum over nodes)
        * temporal smoothness of transport contribution
      (see train script for exact loss).
    """

    def __init__(
        self,
        in_dim: int,
        met_dim: int,
        out_dim: int = 2,
        num_nodes: int = 228,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        zero_init_heads: bool = True,
        enable_lid: bool = True,
        enable_std: bool = True,
        enable_tad: bool = True,
        enable_dynamic_std_gate: bool = False,
        dynamic_gate_hidden_dim: int = 32,
        dynamic_gate_temperature: float = 1.0,
        dynamic_gate_blend_init: float = -3.0,
        enable_wind_direction_graph: bool = False,
        wind_direction_blend_init: float = -2.0,
        wind_direction_scale_init: float = 2.0,
        transport_direction: str = "current",
        enable_wind_history_gate: bool = False,
        wind_history_gate_hidden_dim: int = 16,
        wind_history_gate_init_bias: float = 2.0,
        wind_history_gate_residual_init: float = 0.0,
        enable_light_dynamic_graph: bool = False,
        light_dynamic_hidden_dim: int = 16,
        light_dynamic_blend_init: float = -3.0,
        light_dynamic_scale_init: float = 1.0,
        enable_adaptive_prior_embedding: bool = False,
        node_emb_dim: int = 8,
        time_emb_dim: int = 8,
        max_time_steps: int = 256,
        adaptive_prior_blend_init: float = -2.0,
        enable_structured_prop_bias: bool = False,
        structured_prop_dim: int = 8,
        structured_prop_scale_init: float = 1.0,
        enable_dual_branch_fusion: bool = False,
        dual_branch_fusion_init: float = -2.0,
        enable_dual_branch_residual: bool = False,
        residual_hidden_dim: int = 16,
        residual_blend_init: float = -3.0,
        residual_scale_init: float = 1.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.met_dim = met_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.enable_lid = enable_lid
        self.enable_std = enable_std
        self.enable_tad = enable_tad
        self.enable_dynamic_std_gate = enable_dynamic_std_gate
        self.dynamic_gate_temperature = float(dynamic_gate_temperature)
        self.enable_wind_direction_graph = enable_wind_direction_graph
        if transport_direction not in {"current", "reverse"}:
            raise ValueError("transport_direction must be one of: current, reverse")
        self.transport_direction = str(transport_direction)
        self.enable_wind_history_gate = enable_wind_history_gate
        self.enable_light_dynamic_graph = enable_light_dynamic_graph
        self.enable_adaptive_prior_embedding = enable_adaptive_prior_embedding
        self.enable_structured_prop_bias = enable_structured_prop_bias
        self.enable_dual_branch_fusion = enable_dual_branch_fusion
        self.enable_dual_branch_residual = enable_dual_branch_residual
        self.max_time_steps = int(max_time_steps)

        # Embedding: [x_prev, met] -> hidden
        self.embed = nn.Linear(in_dim + met_dim, hidden_dim)

        if self.enable_adaptive_prior_embedding:
            self.node_embedding = nn.Embedding(num_nodes, node_emb_dim)
            self.time_embedding = nn.Embedding(self.max_time_steps, time_emb_dim)
            self.adaptive_prior_proj = nn.Linear(node_emb_dim + time_emb_dim, hidden_dim)
            self.adaptive_prior_blend_raw = nn.Parameter(torch.tensor(float(adaptive_prior_blend_init)))
        else:
            self.node_embedding = None
            self.time_embedding = None
            self.adaptive_prior_proj = None
            self.adaptive_prior_blend_raw = None

        if self.enable_structured_prop_bias:
            self.struct_src_embedding = nn.Embedding(num_nodes, structured_prop_dim)
            self.struct_dst_embedding = nn.Embedding(num_nodes, structured_prop_dim)
            self.struct_time_embedding = nn.Embedding(self.max_time_steps, structured_prop_dim)
            self.structured_prop_scale = nn.Parameter(torch.tensor(float(structured_prop_scale_init)))
            self.structured_prop_dim = structured_prop_dim
        else:
            self.struct_src_embedding = None
            self.struct_dst_embedding = None
            self.struct_time_embedding = None
            self.structured_prop_scale = None
            self.structured_prop_dim = structured_prop_dim

        if self.enable_dual_branch_fusion:
            self.dual_branch_fusion_raw = nn.Parameter(torch.tensor(float(dual_branch_fusion_init)))
        else:
            self.dual_branch_fusion_raw = None

        # LID: local interactions
        self.lid = MLPBlock(hidden_dim, hidden_mult=4, dropout=dropout)

        # STD: message passing on graph
        self.std_proj = nn.Linear(hidden_dim, hidden_dim)
        # Extract transport contribution for DIC.
        # We intentionally read out from the SAME hidden message that is added back to H,
        # so DIC gradients also affect the prediction pathway (matching Algorithm 1 in the paper).
        self.std_transport_readout = nn.Linear(hidden_dim, out_dim)
        if self.enable_dynamic_std_gate:
            gate_in_dim = hidden_dim + met_dim
            self.dynamic_gate_q = nn.Linear(gate_in_dim, dynamic_gate_hidden_dim)
            self.dynamic_gate_k = nn.Linear(gate_in_dim, dynamic_gate_hidden_dim)
            self.dynamic_gate_blend_raw = nn.Parameter(torch.tensor(float(dynamic_gate_blend_init)))
        else:
            self.dynamic_gate_q = None
            self.dynamic_gate_k = None
            self.dynamic_gate_blend_raw = None

        if self.enable_wind_direction_graph:
            self.wind_direction_blend_raw = nn.Parameter(torch.tensor(float(wind_direction_blend_init)))
            self.wind_direction_scale = nn.Parameter(torch.tensor(float(wind_direction_scale_init)))
            if self.enable_wind_history_gate:
                # Node features: mean_u, mean_v, mean_speed, std_speed, max_speed, direction_stability.
                # Pair input: src, dst, abs(src-dst), graph-level wind context, edge direction.
                gate_in_dim = 6 * 4 + 2
                self.wind_history_gate = nn.Sequential(
                    nn.Linear(gate_in_dim, wind_history_gate_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(wind_history_gate_hidden_dim, 1),
                )
                nn.init.constant_(self.wind_history_gate[-1].bias, float(wind_history_gate_init_bias))
                self.wind_history_gate_residual_gamma = nn.Parameter(torch.tensor(float(wind_history_gate_residual_init)))
            else:
                self.wind_history_gate = None
                self.wind_history_gate_residual_gamma = None
        else:
            self.wind_direction_blend_raw = None
            self.wind_direction_scale = None
            self.wind_history_gate = None
            self.wind_history_gate_residual_gamma = None

        if self.enable_light_dynamic_graph:
            light_in_dim = hidden_dim + met_dim
            self.light_dynamic_src = nn.Sequential(
                nn.Linear(light_in_dim, light_dynamic_hidden_dim),
                nn.SiLU(),
                nn.Linear(light_dynamic_hidden_dim, 1),
            )
            self.light_dynamic_dst = nn.Sequential(
                nn.Linear(light_in_dim, light_dynamic_hidden_dim),
                nn.SiLU(),
                nn.Linear(light_dynamic_hidden_dim, 1),
            )
            self.light_dynamic_blend_raw = nn.Parameter(torch.tensor(float(light_dynamic_blend_init)))
            self.light_dynamic_scale = nn.Parameter(torch.tensor(float(light_dynamic_scale_init)))
        else:
            self.light_dynamic_src = None
            self.light_dynamic_dst = None
            self.light_dynamic_blend_raw = None
            self.light_dynamic_scale = None

        if self.enable_dual_branch_residual:
            residual_in_dim = hidden_dim + met_dim
            self.residual_src = nn.Sequential(
                nn.Linear(residual_in_dim, residual_hidden_dim),
                nn.SiLU(),
                nn.Linear(residual_hidden_dim, 1),
            )
            self.residual_dst = nn.Sequential(
                nn.Linear(residual_in_dim, residual_hidden_dim),
                nn.SiLU(),
                nn.Linear(residual_hidden_dim, 1),
            )
            self.residual_blend_raw = nn.Parameter(torch.tensor(float(residual_blend_init)))
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        else:
            self.residual_src = None
            self.residual_dst = None
            self.residual_blend_raw = None
            self.residual_scale = None

        # TAD: temporal accumulation (GRU per node)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Readout: hidden -> delta_x
        self.readout = nn.Linear(hidden_dim, out_dim)

        # Important for long rollout stability:
        # initialize heads to predict zero delta / zero transport at start,
        # so the initial behavior is a persistence baseline rather than exploding random walk.
        if zero_init_heads:
            nn.init.zeros_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)
            # NOTE: Do NOT zero-init std_transport_readout.
            # If transport head starts exactly at 0, DIC loss can become identically 0 and provide no gradient,
            # causing the DIC term to be effectively ignored during training.

    def _transport_matmul(self, gate: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transport_direction == "reverse":
            gate = gate / (gate.sum(dim=-2, keepdim=True) + 1e-6)
            return gate, torch.matmul(gate.transpose(-2, -1), h)
        gate = gate / (gate.sum(dim=-1, keepdim=True) + 1e-6)
        return gate, torch.matmul(gate, h)

    def _wind_history_reliability(
        self,
        met_all: torch.Tensor,
        hist_len: int,
        edge_dir_dense: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.wind_history_gate is None or self.met_dim < 2:
            return None

        hist_len = max(1, min(int(hist_len), int(met_all.shape[1])))
        wind_hist = met_all[:, :hist_len, :, :2]  # (B,T,N,2)
        speed = torch.norm(wind_hist, dim=-1, keepdim=True).clamp_min(1e-6)
        wind_unit = wind_hist / speed

        uv_mean = wind_hist.mean(dim=1)
        speed_mean = speed.mean(dim=1)
        speed_std = speed.std(dim=1, unbiased=False)
        speed_max = speed.amax(dim=1)
        direction_stability = torch.norm(wind_unit.mean(dim=1), dim=-1, keepdim=True)
        node_feat = torch.cat([uv_mean, speed_mean, speed_std, speed_max, direction_stability], dim=-1)

        src = node_feat.unsqueeze(2).expand(-1, -1, node_feat.shape[1], -1)
        dst = node_feat.unsqueeze(1).expand(-1, node_feat.shape[1], -1, -1)
        diff = torch.abs(src - dst)
        global_wind = node_feat.mean(dim=1, keepdim=True).unsqueeze(2).expand(-1, node_feat.shape[1], node_feat.shape[1], -1)
        edge_dir = edge_dir_dense.unsqueeze(0).expand(node_feat.shape[0], -1, -1, -1)
        pair_feat = torch.cat([src, dst, diff, global_wind, edge_dir], dim=-1)
        return torch.sigmoid(self.wind_history_gate(pair_feat).squeeze(-1))

    def _apply_wind_history_reliability(
        self,
        alignment: torch.Tensor,
        reliability: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if reliability is None or self.wind_history_gate_residual_gamma is None:
            return alignment
        gamma = torch.tanh(self.wind_history_gate_residual_gamma)
        modulation = 1.0 + gamma * (reliability - 0.5)
        return alignment * modulation

    def forward(
        self,
        x_hist: torch.Tensor,
        met_all: torch.Tensor,
        graph: GraphData,
        teacher_forcing: bool = True,
        x_future_gt: Optional[torch.Tensor] = None,
        future_teacher_forcing_prob: float = 0.0,
    ) -> RawinAirOutput:
        """
        x_hist: (B, T_hist, N, C_out)
        met_all: (B, T_hist + T_pred, N, met_dim)
        x_future_gt: optional (B, T_pred, N, C_out) for teacher forcing warm-up behavior
        future_teacher_forcing_prob: during prediction phase (t >= T_hist), probability of feeding ground-truth
          future value (scheduled sampling) instead of model prediction. Set to 0.0 to disable.

        Returns:
          y_pred: predictions for future horizon (B, T_pred, N, C_out)
          transport_pred: extracted transport contribution for future horizon (B, T_pred, N, C_out)
        """
        device = x_hist.device
        # Prefer dense operator (N=228) for speed; sparse+python-loop across batch is very slow.
        ltilde_dense = graph.ltilde_dense.to(device)
        adj_dense = graph.adj_dense.to(device)
        eye = torch.eye(graph.num_nodes, device=device, dtype=ltilde_dense.dtype)
        dyn_mask = (adj_dense > 0).to(ltilde_dense.dtype) + eye
        dyn_mask = (dyn_mask > 0).to(ltilde_dense.dtype)
        edge_dir_dense = graph.edge_dir_dense.to(device)

        b, t_hist, n, c_out = x_hist.shape
        t_total = met_all.shape[1]
        assert met_all.shape[:3] == (b, t_total, n)
        assert met_all.shape[3] == self.met_dim
        assert c_out == self.out_dim
        assert t_total >= t_hist + 1, "met_all must cover at least history and 1 future step"

        t_pred = t_total - t_hist
        if x_future_gt is not None:
            assert x_future_gt.shape == (b, t_pred, n, c_out), f"x_future_gt shape {x_future_gt.shape} != {(b, t_pred, n, c_out)}"
        wind_history_reliability = self._wind_history_reliability(met_all, t_hist, edge_dir_dense)

        # Hidden state per node (B*N, H) for GRU (only used when enable_tad=True)
        z = torch.zeros((b * n, self.hidden_dim), device=device, dtype=x_hist.dtype)

        # Start from first observed as x_prev (teacher forcing will override anyway).
        x_prev = x_hist[:, 0]  # (B,N,C)

        y_preds = []
        g_preds = []

        # Iterate over time steps 1..t_total-1, producing x_t from x_{t-1}
        for t in range(1, t_total):
            met_t = met_all[:, t]  # (B,N,M)
            h = self.embed(torch.cat([x_prev, met_t], dim=-1))  # (B,N,H)
            if self.enable_adaptive_prior_embedding and self.adaptive_prior_proj is not None:
                node_ids = torch.arange(n, device=device, dtype=torch.long)
                node_emb = self.node_embedding(node_ids).unsqueeze(0).expand(b, -1, -1)
                time_id = min(t, self.max_time_steps - 1)
                time_emb = self.time_embedding(torch.tensor(time_id, device=device, dtype=torch.long))
                time_emb = time_emb.view(1, 1, -1).expand(b, n, -1)
                prior = self.adaptive_prior_proj(torch.cat([node_emb, time_emb], dim=-1))
                prior_blend = torch.sigmoid(self.adaptive_prior_blend_raw)
                h = h + prior_blend * prior

            # LID (local)
            if self.enable_lid:
                h = self.lid(h)

            # STD (spatial): M = Ltilde @ h
            if self.enable_std:
                m_static = torch.matmul(ltilde_dense, h)  # (B,N,H)
                m = m_static
                if self.enable_dynamic_std_gate and self.dynamic_gate_q is not None:
                    gate_input = torch.cat([h, met_t], dim=-1)  # (B,N,H+M)
                    q = self.dynamic_gate_q(gate_input)
                    k = self.dynamic_gate_k(gate_input)
                    logits = torch.matmul(q, k.transpose(-2, -1))
                    logits = logits / max(1e-6, float(self.dynamic_gate_temperature))
                    logits = logits.masked_fill(dyn_mask.unsqueeze(0) <= 0, -1e9)
                    attn = torch.softmax(logits, dim=-1)
                    attn = attn * dyn_mask.unsqueeze(0)
                    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
                    m_dynamic = torch.matmul(attn, h)
                    blend = torch.sigmoid(self.dynamic_gate_blend_raw)
                    m = (1.0 - blend) * m_static + blend * m_dynamic
                if (
                    self.enable_dual_branch_fusion
                    and self.dual_branch_fusion_raw is not None
                    and self.enable_wind_direction_graph
                    and self.met_dim >= 2
                    and self.enable_structured_prop_bias
                    and self.structured_prop_scale is not None
                ):
                    node_ids = torch.arange(n, device=device, dtype=torch.long)
                    time_id = min(t, self.max_time_steps - 1)

                    src_emb = self.struct_src_embedding(node_ids)  # (N,D)
                    dst_emb = self.struct_dst_embedding(node_ids)  # (N,D)
                    time_emb = self.struct_time_embedding(torch.tensor(time_id, device=device, dtype=torch.long))  # (D,)
                    src_cond = src_emb + time_emb.unsqueeze(0)  # (N,D)
                    pair_bias = torch.matmul(src_cond, dst_emb.transpose(0, 1))
                    pair_bias = pair_bias / max(1.0, float(self.structured_prop_dim) ** 0.5)
                    pair_bias = torch.tanh(pair_bias).unsqueeze(0)  # (1,N,N)

                    struct_score = self.structured_prop_scale * pair_bias
                    struct_gate = torch.sigmoid(struct_score)
                    struct_gate = struct_gate * adj_dense.unsqueeze(0)
                    struct_gate = struct_gate + eye.unsqueeze(0)
                    struct_gate, m_struct = self._transport_matmul(struct_gate, h)

                    wind_vec = met_t[..., :2]
                    wind_norm = torch.norm(wind_vec, dim=-1, keepdim=True).clamp_min(1e-6)
                    wind_unit = wind_vec / wind_norm  # (B,N,2)
                    alignment = (wind_unit.unsqueeze(2) * edge_dir_dense.unsqueeze(0)).sum(dim=-1)
                    alignment = self._apply_wind_history_reliability(alignment, wind_history_reliability)
                    wind_score = self.wind_direction_scale * alignment
                    wind_gate = torch.sigmoid(wind_score)
                    wind_gate = wind_gate * adj_dense.unsqueeze(0)
                    wind_gate = wind_gate + eye.unsqueeze(0)
                    wind_gate, m_wind = self._transport_matmul(wind_gate, h)

                    beta = torch.sigmoid(self.dual_branch_fusion_raw)
                    m = (1.0 - beta) * m_struct + beta * m_wind
                elif self.enable_wind_direction_graph and self.wind_direction_blend_raw is not None and self.met_dim >= 2:
                    wind_vec = met_t[..., :2]
                    wind_norm = torch.norm(wind_vec, dim=-1, keepdim=True).clamp_min(1e-6)
                    wind_unit = wind_vec / wind_norm  # (B,N,2)
                    alignment = (wind_unit.unsqueeze(2) * edge_dir_dense.unsqueeze(0)).sum(dim=-1)
                    alignment = self._apply_wind_history_reliability(alignment, wind_history_reliability)
                    wind_score = self.wind_direction_scale * alignment
                    if self.enable_structured_prop_bias and self.structured_prop_scale is not None:
                        node_ids = torch.arange(n, device=device, dtype=torch.long)
                        src_emb = self.struct_src_embedding(node_ids)  # (N,D)
                        dst_emb = self.struct_dst_embedding(node_ids)  # (N,D)
                        time_id = min(t, self.max_time_steps - 1)
                        time_emb = self.struct_time_embedding(torch.tensor(time_id, device=device, dtype=torch.long))  # (D,)
                        src_cond = src_emb + time_emb.unsqueeze(0)  # (N,D)
                        pair_bias = torch.matmul(src_cond, dst_emb.transpose(0, 1))
                        pair_bias = pair_bias / max(1.0, float(self.structured_prop_dim) ** 0.5)
                        pair_bias = torch.tanh(pair_bias).unsqueeze(0)  # (1,N,N)
                        wind_score = wind_score + self.structured_prop_scale * pair_bias
                    wind_gate = torch.sigmoid(wind_score)
                    wind_gate = wind_gate * adj_dense.unsqueeze(0)
                    wind_gate = wind_gate + eye.unsqueeze(0)
                    wind_gate, m_wind = self._transport_matmul(wind_gate, h)
                    wind_blend = torch.sigmoid(self.wind_direction_blend_raw)
                    m = (1.0 - wind_blend) * m + wind_blend * m_wind
                if self.enable_light_dynamic_graph and self.light_dynamic_blend_raw is not None:
                    light_input = torch.cat([h, met_t], dim=-1)  # (B,N,H+M)
                    src_score = self.light_dynamic_src(light_input).squeeze(-1)  # (B,N)
                    dst_score = self.light_dynamic_dst(light_input).squeeze(-1)  # (B,N)
                    pair_score = src_score.unsqueeze(-1) + dst_score.unsqueeze(-2)  # (B,N,N)
                    light_gate = torch.sigmoid(self.light_dynamic_scale * pair_score)
                    light_gate = light_gate * adj_dense.unsqueeze(0)
                    light_gate = light_gate + eye.unsqueeze(0)
                    light_gate = light_gate / (light_gate.sum(dim=-1, keepdim=True) + 1e-6)
                    m_light = torch.matmul(light_gate, h)
                    light_blend = torch.sigmoid(self.light_dynamic_blend_raw)
                    m = (1.0 - light_blend) * m + light_blend * m_light
                if self.enable_dual_branch_residual and self.residual_blend_raw is not None:
                    residual_input = torch.cat([h, met_t], dim=-1)  # (B,N,H+M)
                    residual_src = self.residual_src(residual_input).squeeze(-1)  # (B,N)
                    residual_dst = self.residual_dst(residual_input).squeeze(-1)  # (B,N)
                    residual_score = residual_src.unsqueeze(-1) + residual_dst.unsqueeze(-2)  # (B,N,N)
                    residual_gate = torch.sigmoid(self.residual_scale * residual_score)
                    residual_gate = residual_gate * adj_dense.unsqueeze(0)
                    residual_gate = residual_gate / (residual_gate.sum(dim=-1, keepdim=True) + 1e-6)
                    m_residual = torch.matmul(residual_gate, h)
                    residual_blend = torch.sigmoid(self.residual_blend_raw)
                    m = m + residual_blend * m_residual
                m_hidden = self.std_proj(m)  # (B,N,H)  message injected into hidden state
                h = h + m_hidden
                transport = self.std_transport_readout(m_hidden)  # (B,N,C_out)
            else:
                transport = torch.zeros((b, n, self.out_dim), device=device, dtype=h.dtype)

            # TAD (temporal): GRU per node
            if self.enable_tad:
                h2 = h.reshape(b * n, self.hidden_dim)
                z = self.gru(h2, z)
                h = h + z.reshape(b, n, self.hidden_dim)

            # Readout delta and residual update
            delta = self.readout(h)  # (B,N,C_out)
            x_hat = x_prev + delta

            # During warm-up, keep state aligned to ground truth
            if teacher_forcing and t < t_hist:
                x_prev = x_hist[:, t]
            else:
                # Scheduled sampling (prediction phase): optionally feed future ground truth.
                # This helps reduce exposure bias for long-horizon rollouts.
                if (
                    teacher_forcing
                    and (x_future_gt is not None)
                    and (future_teacher_forcing_prob > 0.0)
                    and (t >= t_hist)
                ):
                    # index into future horizon: t=t_hist..t_total-1 -> k=0..t_pred-1
                    k = t - t_hist
                    if future_teacher_forcing_prob >= 1.0:
                        x_prev = x_future_gt[:, k]
                    else:
                        use_gt = (torch.rand((b, 1, 1), device=device, dtype=torch.float32) < float(future_teacher_forcing_prob))
                        x_prev = torch.where(use_gt, x_future_gt[:, k], x_hat)
                else:
                    x_prev = x_hat

            # Collect predictions only for future horizon (t >= t_hist)
            if t >= t_hist:
                y_preds.append(x_hat)
                g_preds.append(transport)

        y_pred = torch.stack(y_preds, dim=1)  # (B,T_pred,N,C)
        transport_pred = torch.stack(g_preds, dim=1)  # (B,T_pred,N,C)
        assert y_pred.shape[1] == t_pred
        return RawinAirOutput(y_pred=y_pred, transport_pred=transport_pred)
