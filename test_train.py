import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import softmax, degree
from torch_scatter import scatter_add, scatter_max, scatter_mean
import networkx as nx
from sklearn.ensemble import RandomForestClassifier


# =============================================================================
# Shared sub-modules
# =============================================================================

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, use_bn=True, dropout=0.0):
        super().__init__()
        h = hidden_dim or out_dim
        layers = [
            nn.Linear(in_dim, h),
            nn.BatchNorm1d(h) if use_bn else nn.Identity(),
            nn.ELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(h, out_dim),
                   nn.BatchNorm1d(out_dim) if use_bn else nn.Identity(),
                   nn.ELU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GATLayer(nn.Module):
    """Multi-head GAT matching Eq.(3)-(4) in the paper."""
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert out_dim % num_heads == 0
        self.H, self.d = num_heads, out_dim // num_heads
        self.W = nn.Parameter(torch.empty(num_heads, in_dim, self.d))
        self.a = nn.Parameter(torch.empty(num_heads, 2 * self.d, 1))
        self.proj = nn.Linear(out_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, edge_index):
        N = x.size(0)
        src, dst = edge_index
        heads = []
        for h in range(self.H):
            hx = x @ self.W[h]                              # (N, d)
            ef = torch.cat([hx[src], hx[dst]], dim=1)       # (E, 2d)
            e  = F.leaky_relu(ef @ self.a[h], 0.2)          # (E, 1)
            a  = self.drop(softmax(e, dst, num_nodes=N))
            heads.append(scatter_add(hx[src] * a, dst, dim=0, dim_size=N))
        return self.proj(torch.cat(heads, dim=1))


class GCNLayer(nn.Module):
    """Symmetric-normalised GCN matching Eq.(2) in the paper."""
    def __init__(self, in_dim, out_dim, use_bn=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.bn  = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()

    def forward(self, x, edge_index):
        src, dst = edge_index
        N   = x.size(0)
        deg = degree(edge_index[0], N).float().clamp(min=1)
        norm_src = deg[src].pow(-0.5)
        norm_dst = deg[dst].pow(-0.5)
        xw  = self.lin(x)
        msg = xw[src] * (norm_src * norm_dst).unsqueeze(1)
        out = scatter_add(msg, dst, dim=0, dim_size=N)
        return self.bn(F.elu(out))


# =============================================================================
# Phase 1 – Engage  (Eq. 1-5)
# Dual GCN-GAT aggregation with gated fusion and feature resource connectors
# =============================================================================

class FeatureResourceConnector(MessagePassing):
    def __init__(self, dim, use_bn=True):
        super().__init__(aggr='mean')
        self.mlp = MLP(dim, dim, use_bn=use_bn)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return self.mlp(x_j)


class FeatureWeightAggregator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2) / 2)
        self.p = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])

    def forward(self, fn, frn):
        a = F.softmax(self.w, dim=0)
        return a[0] * self.p[0](fn) + a[1] * self.p[1](frn)


class EngageModule(nn.Module):
    """
    Phase 1: Engage
    H_E = GCN(x) + GAT(x)  with gated fusion, feature connectors, and residual.
    Implements Eq.(1-5) of the paper.
    """
    def __init__(self, in_node_dim, in_edge_dim=None, hidden_dim=64,
                 num_heads=4, dropout=0.1, use_edge_features=False,
                 use_batch_norm=True, residual=True):
        super().__init__()
        self.residual = residual
        self.hidden_dim = hidden_dim

        self.node_proj = MLP(in_node_dim, hidden_dim, use_bn=use_batch_norm)
        self.gcn       = GCNLayer(hidden_dim, hidden_dim, use_bn=use_batch_norm)
        self.gat       = GATLayer(hidden_dim, hidden_dim, num_heads, dropout)
        self.connector = FeatureResourceConnector(hidden_dim, use_batch_norm)
        self.aggregator = FeatureWeightAggregator(hidden_dim)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ELU())
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        fn  = self.node_proj(x)
        frn = self.connector(fn, edge_index)
        agg = self.aggregator(fn, frn)

        h_gcn = self.gcn(fn, edge_index)
        h_gat = self.gat(fn, edge_index)

        # Eq.(5): H_E = h_GCN + h_GAT  (additive fusion)
        h_fused = h_gcn + h_gat

        # gated blend: balance structured aggregation vs attention output
        gate = self.gate(torch.cat([agg, h_fused], dim=1))
        h    = gate * agg + (1 - gate) * h_fused
        h    = self.out(h)

        if self.residual and fn.shape == h.shape:
            h = h + fn
        return self.drop(h)


# =============================================================================
# Phase 2 – Enact  (Eq. 6-14)
# Node importance weighting, metapath extraction, adaptive sampling,
# edge intensity modelling, anomaly detection, knowledge application
# =============================================================================

class NodeImportanceWeighting(nn.Module):
    """Eq.(9): W_i = sigmoid(W_w h_i + b_w)"""
    def __init__(self, hidden_dim, num_classes, use_bn=True):
        super().__init__()
        self.prototypes   = nn.Parameter(torch.empty(num_classes, hidden_dim))
        self.weight_gen   = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
            nn.ELU(), nn.Linear(hidden_dim, hidden_dim))
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(), nn.ELU())
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x, labels=None, mask=None):
        if labels is not None and mask is not None and mask.sum() > 0:
            for c in range(self.prototypes.size(0)):
                cm = (labels == c) & mask
                if cm.sum() > 0:
                    self.prototypes.data[c].mul_(0.9).add_(
                        x[cm].mean(0).detach() * 0.1)
        sim = F.normalize(x, 2, 1) @ F.normalize(self.prototypes, 2, 1).t()
        w   = self.weight_gen(sim)
        return self.proj(torch.cat([x, x * w], dim=1))


class MetapathExtractor(nn.Module):
    """
    Eq.(10-11): vectorised 2-hop metapath via scatter_mean.
    F_weight = mean over paths of (prod W_v * h_v_last)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, node_weights=None):
        src, dst = edge_index
        N = x.size(0)
        hop1 = scatter_mean(x[src], dst, dim=0, dim_size=N)
        hop2 = scatter_mean(hop1[src], dst, dim=0, dim_size=N)
        if node_weights is not None:
            w = torch.sigmoid(node_weights).unsqueeze(1)
            meta = w * hop1 + (1 - w) * hop2
        else:
            meta = 0.5 * hop1 + 0.5 * hop2
        norm = meta.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        return self.proj(meta / norm)


class EdgeIntensityModelling(nn.Module):
    """Eq.(13): e_ij = sigmoid(h_i^T W_e h_j)"""
    def __init__(self, hidden_dim, use_bn=True):
        super().__init__()
        self.W_e  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(), nn.ELU())
        self.out  = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(), nn.ELU())

    def forward(self, x, edge_index):
        src, dst = edge_index
        # edge intensity: scalar similarity weighted by W_e (Eq.13)
        e_ij = torch.sigmoid((self.W_e(x[src]) * x[dst]).sum(dim=1, keepdim=True))
        edge_feat = self.proj(torch.cat([x[src], x[dst]], dim=1))
        agg = scatter_add(edge_feat * e_ij, dst, dim=0, dim_size=x.size(0))
        return self.out(torch.cat([x, agg], dim=1))


class AnomalyDetection(nn.Module):
    """Eq.(14): anomaly = ||h_v - mu_c|| based detection + subgraph context"""
    def __init__(self, hidden_dim, use_bn=True):
        super().__init__()
        self.anomaly_proc  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(), nn.ELU())
        self.subgraph_proc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(), nn.ELU())
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(), nn.ELU())

    def forward(self, x, edge_index, labels=None, mask=None, num_classes=7):
        src, dst = edge_index
        diff   = torch.norm(x[src] - x[dst], p=2, dim=1)
        ndiff  = scatter_mean(diff, src, dim=0, dim_size=x.size(0))
        dmin, dmax = ndiff.min(), ndiff.max()
        scores = (ndiff - dmin) / (dmax - dmin + 1e-8)

        if labels is not None and mask is not None and mask.sum() > 0:
            C       = min(num_classes, labels.max().item() + 1)
            protos  = torch.zeros(C, x.size(1), device=x.device)
            for c in range(C):
                cm = (labels == c) & mask
                if cm.sum() > 0:
                    protos[c] = x[cm].mean(0)
            own = protos[labels.clamp(0, C - 1)]
            la  = torch.norm(x - own, p=2, dim=1) * mask.float()
            la_max = la.max()
            if la_max > 0:
                la = la / la_max
                scores[mask] = 0.7 * la[mask] + 0.3 * scores[mask]

        hop1     = scatter_mean(x[src], dst, dim=0, dim_size=x.size(0))
        hop2     = scatter_mean(hop1[src], dst, dim=0, dim_size=x.size(0))
        subgraph = 0.5 * hop1 + 0.3 * hop2 + 0.2 * x

        af = self.anomaly_proc(x) * scores.unsqueeze(1)
        sf = self.subgraph_proc(subgraph)
        return self.proj(torch.cat([x, af, sf], dim=1))


class AdaptiveSampling(nn.Module):
    """Eq.(12): selects nodes by uncertainty and minority membership."""
    def __init__(self, hidden_dim, use_bn=True):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(), nn.ELU())
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(), nn.ELU())

    def forward(self, x, edge_index, labels=None, mask=None, num_classes=7):
        deg  = degree(edge_index[0], x.size(0)).float()
        ndeg = deg / (deg.max() + 1e-8)
        src, dst = edge_index

        if labels is None or mask is None or mask.sum() == 0:
            sim  = (F.normalize(x[src], 2, 1) * F.normalize(x[dst], 2, 1)).sum(1)
            cnt  = scatter_add(torch.ones_like(sim), src, dim=0, dim_size=x.size(0))
            nsim = scatter_add(sim, src, dim=0, dim_size=x.size(0)) / (cnt + 1e-8)
            scores = 0.7 * (1 - nsim) + 0.3 * ndeg
        else:
            C      = min(num_classes, labels.max().item() + 1)
            lf     = x[mask]
            ll     = labels[mask]
            protos = torch.zeros(C, x.size(1), device=x.device)
            for c in range(C):
                cm = (ll == c)
                if cm.sum() > 0:
                    protos[c] = lf[cm].mean(0)
            dists = torch.cdist(x, protos)
            own_d = dists[torch.arange(x.size(0), device=x.device),
                          labels.clamp(0, C - 1)]
            inf_m = torch.zeros_like(dists)
            inf_m.scatter_(1, labels.clamp(0, C - 1).unsqueeze(1), float('inf'))
            wrong = (dists + inf_m).min(1).values
            bscore = own_d / (wrong + 1e-8)
            scores = 0.5 * bscore + 0.3 * ndeg + 0.2 * (1 - mask.float())

        t   = self.transform(x) * scores.unsqueeze(1)
        agg = scatter_mean(t[src], dst, dim=0, dim_size=x.size(0))
        return self.out(0.7 * t + 0.3 * agg)


class EnactFeedback(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.Sigmoid())
        self.proj = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, combined, component0, imbalance, metapath):
        wi  = component0 * imbalance.unsqueeze(1)
        inp = torch.cat([combined, wi, metapath], dim=1)
        g   = self.gate(inp)
        return combined * (1 - g) + self.proj(inp) * g


class EnactModule(nn.Module):
    """
    Phase 2: Enact — Eq.(6-14)
    Node importance weighting (W_i), metapath extraction (M),
    adaptive sampling (S), edge intensity e_ij, anomaly detection (C),
    combined via learned attention into H_Enact = Eq.(14).
    """
    def __init__(self, hidden_dim, num_classes, dropout=0.1, use_batch_norm=True):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_classes = num_classes

        self.node_importance = NodeImportanceWeighting(hidden_dim, num_classes, use_batch_norm)
        self.metapath        = MetapathExtractor(hidden_dim)
        self.sampling        = AdaptiveSampling(hidden_dim, use_batch_norm)
        self.edge_intensity  = EdgeIntensityModelling(hidden_dim, use_batch_norm)
        self.anomaly         = AnomalyDetection(hidden_dim, use_batch_norm)

        # knowledge application from prototypes
        self.knowledge_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.knowledge_out   = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(), nn.ELU())

        # attention over 5 components (Eq.14 generalisation)
        self.component_attn  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 5))

        # imbalance-aware scaling
        self.imbalance_scale = nn.Linear(1, hidden_dim, bias=False)
        self.feedback        = EnactFeedback(hidden_dim)

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ELU())
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, partial_labels=None, mask=None, batch=None):
        ni   = self.node_importance(x, partial_labels, mask)

        # node-level importance weights W_i (scalar)
        ni_w = torch.sigmoid(ni.mean(1))

        meta = self.metapath(x, edge_index, node_weights=ni_w)
        samp = self.sampling(x, edge_index, partial_labels, mask, self.num_classes)
        ei   = self.edge_intensity(x, edge_index)
        anom = self.anomaly(x, edge_index, partial_labels, mask, self.num_classes)

        # knowledge from class prototypes
        with torch.no_grad():
            k_proto = self.node_importance.prototypes.detach()
        kp     = self.knowledge_proj(k_proto)
        # lightweight knowledge application: mean prototype weighted by similarity
        sim    = F.normalize(x, 2, 1) @ F.normalize(kp, 2, 1).t()   # (N, C)
        k_feat = self.knowledge_out(torch.cat([x, sim @ kp], dim=1))

        components = [ni, samp, ei, anom, k_feat]
        aw = F.softmax(self.component_attn(x), dim=1)
        h  = sum(aw[:, i:i+1] * c for i, c in enumerate(components))

        # imbalance-aware reweighting
        imb = self._imbalance_scores(x, edge_index, partial_labels, mask)
        h   = self.feedback(h, ni, imb, meta)

        return self.drop(self.out(h))

    def _imbalance_scores(self, x, edge_index, labels, mask):
        scores = torch.ones(x.size(0), device=x.device)
        if labels is not None and mask is not None and mask.sum() > 0:
            C = min(self.num_classes, labels.max().item() + 1)
            cnt = torch.zeros(C, device=x.device)
            for c in range(C):
                cnt[c] = ((labels == c) & mask).float().sum()
            if cnt.sum() > 0:
                imb = 1 - cnt / cnt.sum()
                scores[mask] = imb[labels[mask].clamp(0, C - 1)]
        return scores


# =============================================================================
# Phase 3 – Embed  (Eq. 15-23)
# Memory-augmented update (M), gradient-based historical updates,
# knowledge graph integration, RL-style value network, learning strategies
# =============================================================================

class MemoryAugmentedUpdate(nn.Module):
    """
    Eq.(17): h^m = h + softmax(h^T M) M^T
    Differentiable memory with EMA write (Algorithm 1).
    """
    def __init__(self, hidden_dim, num_slots=8, use_bn=True):
        super().__init__()
        self.M_keys  = nn.Parameter(torch.empty(num_slots, hidden_dim))
        self.M_vals  = nn.Parameter(torch.empty(num_slots, hidden_dim))
        self.ctrl    = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(), nn.Sigmoid())
        self.proc    = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(), nn.ELU())
        nn.init.xavier_uniform_(self.M_keys)
        nn.init.xavier_uniform_(self.M_vals)

    def forward(self, x):
        q    = F.normalize(x @ self.M_keys.t(), dim=1)   # (N, slots)
        att  = F.softmax(q, dim=1)
        r    = att @ self.M_vals                           # (N, H) — Eq.(17)
        with torch.no_grad():
            cont = F.normalize(att.t(), p=1, dim=1)
            self.M_vals.data.mul_(0.95).add_(0.05 * cont @ x)
        c   = self.ctrl(torch.cat([x, r], dim=1))
        out = self.proc(torch.cat([x, r], dim=1))
        return c * out + (1 - c) * x


class KnowledgeGraphIntegration(nn.Module):
    """1-hop + 2-hop neighbourhood aggregation matching Eq.(15) structure."""
    def __init__(self, hidden_dim, use_bn=True):
        super().__init__()
        self.ext  = MLP(hidden_dim, hidden_dim, use_bn=use_bn)
        self.inte = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(), nn.ELU())

    def forward(self, x, edge_index):
        src, dst = edge_index
        k    = self.ext(x)
        h1   = scatter_mean(k[src], dst, dim=0, dim_size=x.size(0))
        h2   = scatter_mean(h1[src], dst, dim=0, dim_size=x.size(0))
        return self.inte(torch.cat([x, h1, h2], dim=1))


class GradientHistoricalUpdate(nn.Module):
    """
    Eq.(18): four strategies — cross-task transfer, historical gradient,
    error-driven feedback, contextual modulation (Algorithm 2).
    """
    def __init__(self, hidden_dim, num_classes, use_bn=True):
        super().__init__()
        self.W_transfer = nn.Linear(hidden_dim, hidden_dim)
        self.value_net  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2) if use_bn else nn.Identity(),
            nn.ELU(), nn.Linear(hidden_dim // 2, 1))
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
            nn.ELU(), nn.Linear(hidden_dim, num_classes))
        self.enhance = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
            nn.ELU(), nn.Linear(hidden_dim, hidden_dim))
        # context modulation
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.strategy_sel = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2) if use_bn else nn.Identity(),
            nn.ELU(), nn.Linear(hidden_dim // 2, 3), nn.Softmax(dim=1))
        self.strategies   = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, use_bn=use_bn),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2) if use_bn else nn.Identity(),
                nn.ELU(), nn.Linear(hidden_dim // 2, hidden_dim)),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2) if use_bn else nn.Identity(),
                nn.ELU(), nn.Linear(hidden_dim * 2, hidden_dim))
        ])
        self.integrate = MLP(hidden_dim, hidden_dim, use_bn=use_bn)

    def forward(self, x, rewards):
        # cross-task transfer (h^t)
        h_t = self.W_transfer(x)
        # historical gradient update (h^h) via advantage signal
        val = self.value_net(x)
        adv = rewards.unsqueeze(1) - val
        h_h = x + self.enhance(torch.cat([x, adv], dim=1))
        # contextual modulation (h^c)
        cg  = self.context_gate(torch.cat([h_t, h_h], dim=1))
        h_c = x * cg
        # strategy-weighted combination (h^f equivalent)
        base   = h_t + h_h + h_c
        sw     = self.strategy_sel(base)
        h_f    = sum(self.strategies[i](base) * sw[:, i:i+1] for i in range(3))
        return self.integrate(h_f)


class ReflectiveLearning(nn.Module):
    """Meta-learning: reflect on all knowledge components to self-adjust."""
    def __init__(self, hidden_dim, n_components=4, use_bn=True):
        super().__init__()
        self.reflect = nn.Sequential(
            nn.Linear(hidden_dim * (n_components + 1), hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2) if use_bn else nn.Identity(),
            nn.ELU(), nn.Linear(hidden_dim * 2, hidden_dim))
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2) if use_bn else nn.Identity(),
            nn.ELU(), nn.Linear(hidden_dim // 2, hidden_dim), nn.Sigmoid())

    def forward(self, x, components):
        r  = self.reflect(torch.cat([x] + components, dim=1))
        g  = self.gate(torch.cat([x, r], dim=1))
        return g * r + (1 - g) * x


class EmbedModule(nn.Module):
    """
    Phase 3: Embed — Eq.(15-22)
    Memory-augmented update, knowledge graph integration, gradient-historical
    updates (4 strategies), reflective learning.
    H_Embed = g + a + m + t + h + f + x  (Eq.22)
    Returns (embeddings, logits).
    """
    def __init__(self, hidden_dim, num_classes, dropout=0.1, use_batch_norm=True):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_classes = num_classes

        self.memory     = MemoryAugmentedUpdate(hidden_dim, use_bn=use_batch_norm)
        self.knowledge  = KnowledgeGraphIntegration(hidden_dim, use_batch_norm)
        self.grad_hist  = GradientHistoricalUpdate(hidden_dim, num_classes, use_batch_norm)
        self.reflective = ReflectiveLearning(hidden_dim, 4, use_batch_norm)
        self.attn       = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, 4))
        self.out  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ELU())
        self.cls  = nn.Linear(hidden_dim, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, partial_labels=None, mask=None, batch=None):
        kg   = self.knowledge(x, edge_index)
        mem  = self.memory(x)
        rwd  = self._rewards(x, edge_index, partial_labels, mask)
        gh   = self.grad_hist(x, rwd)
        strat = kg  # strategy component (knowledge graph = transfer source)

        components = [kg, mem, gh, strat]
        aw = F.softmax(self.attn(x), dim=1)
        h  = sum(aw[:, i:i+1] * c for i, c in enumerate(components))

        # Eq.(22): sum all adapted representations + reflective
        refl = self.reflective(x, components)
        h    = h + 0.3 * refl

        emb    = self.drop(self.out(h))
        logits = self.cls(emb)
        return emb, logits

    def _rewards(self, x, edge_index, labels, mask):
        N = x.size(0)
        r = torch.zeros(N, device=x.device)
        if labels is not None and mask is not None and mask.sum() > 0:
            with torch.no_grad():
                preds = self.cls(x).argmax(1)
                r[(preds == labels) & mask] = 1.0
            src, dst = edge_index
            for _ in range(2):
                prop, _ = scatter_max(0.7 * r[src], dst, dim=0, dim_size=N)
                r = torch.max(r, prop)
        else:
            deg = degree(edge_index[0], N).float()
            r   = deg / (deg.max() + 1e-8)
        return r


# =============================================================================
# CL3EClassifier  (Eq. 23-29)
# Unified three-phase model with:
#   • gated skip connections between phases
#   • residual additions for gradient flow
#   • phase-specific auxiliary supervision
#   • label-smoothed aux loss
#   • learned ensemble of three logit heads
#   • three-phase curriculum loss (Eq.25-29)
# =============================================================================

class CL3EClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes,
                 dropout=0.5, use_batch_norm=True,
                 aux_weight=0.15, label_smoothing=0.1):
        super().__init__()
        self.aux_weight      = aux_weight
        self.label_smoothing = label_smoothing
        self.hidden_dim      = hidden_dim
        self.num_classes     = num_classes

        self.engage = EngageModule(in_node_dim=in_dim, hidden_dim=hidden_dim,
                                   dropout=dropout, use_batch_norm=use_batch_norm)
        self.enact  = EnactModule(hidden_dim=hidden_dim, num_classes=num_classes,
                                  dropout=dropout, use_batch_norm=use_batch_norm)
        self.embed  = EmbedModule(hidden_dim=hidden_dim, num_classes=num_classes,
                                  dropout=dropout, use_batch_norm=use_batch_norm)

        # gated skip: engage → enact input
        self.gate_e2n = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.Sigmoid())
        # gated blend: [enact, engage] → embed input  (Eq.23)
        self.gate_n2e = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.Sigmoid())

        # auxiliary classification heads on intermediate representations
        self.aux_engage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes))
        self.aux_enact  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes))

        # learned phase ensemble weights (Eq.23)
        self.phase_w = nn.Parameter(torch.tensor([0.2, 0.3, 0.5]))

    def forward(self, x, edge_index, edge_attr=None,
                partial_labels=None, mask=None, batch=None):

        # Phase 1 – Engage
        h_e = self.engage(x, edge_index, edge_attr=edge_attr, batch=batch)

        # Phase 2 – Enact with gated skip from Engage
        g_e2n  = self.gate_e2n(h_e)
        h_n    = self.enact(h_e * g_e2n, edge_index,
                            partial_labels=partial_labels, mask=mask)
        h_n    = h_n + h_e                         # residual

        # Phase 3 – Embed with gated blend of Enact + Engage
        g_n2e  = self.gate_n2e(torch.cat([h_n, h_e], dim=1))
        h_in   = g_n2e * h_n + (1 - g_n2e) * h_e
        emb, logits_embed = self.embed(h_in, edge_index,
                                       partial_labels=partial_labels, mask=mask)

        # Auxiliary heads for intermediate supervision
        logits_e = self.aux_engage(h_e)
        logits_n = self.aux_enact(h_n)

        # Ensemble (Eq.23): H_final = H_E + H_Enact + H_Embed
        pw     = F.softmax(self.phase_w, dim=0)
        logits = pw[0] * logits_e + pw[1] * logits_n + pw[2] * logits_embed

        return emb, logits, logits_e, logits_n

    def aux_loss(self, lg_e, lg_n, targets):
        def _sce(logits, tgt):
            n_cls = logits.size(1)
            s     = self.label_smoothing
            with torch.no_grad():
                soft = torch.zeros_like(logits).scatter_(1, tgt.unsqueeze(1), 1.0)
                soft = soft * (1 - s) + s / n_cls
            return -(soft * F.log_softmax(logits, dim=1)).sum(1).mean()
        return self.aux_weight * 0.5 * (_sce(lg_e, targets) + _sce(lg_n, targets))


# =============================================================================
# CL3E_ModelV1 — GCN-backbone with three-stage attention + curriculum factor
# Now shares EngageModule for feature extraction (integrated architecture)
# =============================================================================

class CL3E_ModelV1(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2,
                 dropout=0.5, attention_type='sigmoid', use_batch_norm=True):
        super().__init__()
        self.num_features   = in_dim
        self.hidden_dim     = hidden_dim
        self.num_classes    = num_classes
        self.num_gcn_layers = num_layers
        self.attention_type = attention_type
        self.dropout_rate   = dropout

        # Phase 1: Engage for initial feature extraction
        self.engage = EngageModule(in_node_dim=in_dim, hidden_dim=hidden_dim,
                                   dropout=dropout, use_batch_norm=use_batch_norm)

        # GCN backbone layers
        self.gcn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.final_gcn  = GCNConv(hidden_dim, num_classes)
        self.dropout    = nn.Dropout(dropout)

        # Three-stage attention (original V1 design)
        lin = nn.Linear
        if attention_type == 'sigmoid':
            self.s1 = lin(hidden_dim, 1)
            self.s2 = lin(hidden_dim, 1)
            self.s3 = lin(hidden_dim, 1)
        else:
            self.s1 = lin(hidden_dim, hidden_dim)
            self.s2 = lin(hidden_dim, hidden_dim)
            self.s3 = lin(hidden_dim, hidden_dim)

        self.curriculum_factor = nn.Parameter(torch.ones(num_classes))

    def forward(self, x, edge_index, partial_labels=None, mask=None, **kw):
        # Engage phase provides initial representation
        x = self.engage(x, edge_index)

        for layer in self.gcn_layers:
            x = self.dropout(F.relu(layer(x, edge_index)))

        r1 = torch.sum(x * torch.sigmoid(self.s1(x)), dim=0)
        r2 = torch.sum(x * torch.sigmoid(self.s2(x)), dim=0)
        r3 = torch.sum(x * torch.sigmoid(self.s3(x)), dim=0)
        combined = r1 + r2 + r3

        out = self.final_gcn(x, edge_index)
        return out, combined

    def curriculum_loss(self, outputs, targets):
        loss = F.cross_entropy(outputs, targets, reduction='none')
        return (loss * self.curriculum_factor[targets]).mean()

    def adjust_curriculum(self, difficulty_scores):
        with torch.no_grad():
            for c, d in difficulty_scores.items():
                if c < self.num_classes:
                    self.curriculum_factor[c] = 2.0 - d
        return self.curriculum_factor.detach().cpu().numpy()


# =============================================================================
# CL3E_ModelV2 — three-phase integration with stage-wise attention
# Engage → GCN backbone (stage attention) → Enact refinement → final output
# Retains V2 curriculum, community, metapath, transfer learning extras
# =============================================================================

class CL3E_ModelV2(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes,
                 graph_data=None, num_nodes=None, model=None,
                 num_layers=2, dropout=0.5,
                 attention_type='sigmoid', use_batch_norm=True,
                 max_metapaths=30):
        super().__init__()
        self.num_features   = in_dim
        self.hidden_dim     = hidden_dim
        self.num_classes    = num_classes
        self.num_gcn_layers = num_layers
        self.attention_type = attention_type
        self.dropout_rate   = dropout
        self.graph_data     = graph_data
        self.num_nodes      = num_nodes or (len(graph_data) if graph_data else 0)
        self.model          = model or RandomForestClassifier(n_estimators=100, random_state=42)

        # Phase 1: Engage
        self.engage = EngageModule(in_node_dim=in_dim, hidden_dim=hidden_dim,
                                   dropout=dropout, use_batch_norm=use_batch_norm)
        # Phase 2: Enact for feature refinement
        self.enact  = EnactModule(hidden_dim=hidden_dim, num_classes=num_classes,
                                  dropout=dropout, use_batch_norm=use_batch_norm)

        # GCN backbone
        self.gcn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.final_gcn  = GCNConv(hidden_dim, num_classes)
        self.dropout    = nn.Dropout(dropout)
        self.curriculum_factor = nn.Parameter(torch.ones(num_classes))

        # Stage-wise attention on intermediate outputs (original V2 design)
        lin = nn.Linear
        if attention_type == 'sigmoid':
            self.s1 = lin(hidden_dim, 1)
            self.s2 = lin(hidden_dim, 1)
            self.s3 = lin(hidden_dim, 1)
        else:
            self.s1 = lin(hidden_dim, hidden_dim)
            self.s2 = lin(hidden_dim, hidden_dim)
            self.s3 = lin(hidden_dim, hidden_dim)

        # V2 extras
        self.node_importance_weights = {}
        self.attention_weights       = {}
        self.metapaths               = []
        self.community_structure     = {}
        self._has_analyzed_structure = False
        self._has_detected_communities = False
        self.knowledge_cache    = {}
        self.rewards            = {}
        self.reward_history     = {}
        self.performance_metrics = {}
        self.transfer_history   = []
        self.adaptation_history = []
        self.feature_importance = {}

    def forward(self, x, edge_index, partial_labels=None, mask=None, **kw):
        # Phase 1 + Phase 2 for richer initial representations
        h_e = self.engage(x, edge_index)
        h_n = self.enact(h_e, edge_index, partial_labels=partial_labels, mask=mask)
        h   = h_n + h_e   # residual

        intermediates = [h]
        for layer in self.gcn_layers:
            h = self.dropout(F.relu(layer(h, edge_index)))
            intermediates.append(h)

        mid = len(intermediates) // 2
        r1  = torch.sum(intermediates[0]  * torch.sigmoid(self.s1(intermediates[0])),  dim=0)
        r2  = torch.sum(intermediates[mid]* torch.sigmoid(self.s2(intermediates[mid])), dim=0)
        r3  = torch.sum(h * torch.sigmoid(self.s3(h)), dim=0)
        combined = r1 + r2 + r3

        for i in range(h.size(0)):
            self.attention_weights[i] = torch.sigmoid(self.s3(h))[i].item()

        out = self.final_gcn(h, edge_index)
        out = out * self.curriculum_factor.unsqueeze(0)
        return out, combined

    def adjust_curriculum(self, difficulty_scores):
        with torch.no_grad():
            for c, d in difficulty_scores.items():
                if c < self.num_classes:
                    self.curriculum_factor[c] = 2.0 - d
        return self.curriculum_factor.detach().cpu().numpy()

    def facilitate_learning(self, adj_matrices):
        self.calculate_node_importance(adj_matrices)
        self.extract_metapaths()
        self._has_analyzed_structure = True
        top5 = sorted(self.node_importance_weights.items(), key=lambda x: x[1], reverse=True)[:5]
        return {'top_nodes': top5, 'metapaths': self.metapaths[:5],
                'attention_weights': dict(sorted(self.attention_weights.items(),
                                                  key=lambda x: x[1], reverse=True)[:5]),
                'structure_analyzed': True}

    def calculate_node_importance(self, adj_matrices):
        if not adj_matrices:
            return {}
        A  = adj_matrices[0]
        dc = torch.sum(A, dim=1)
        ev = torch.matmul(A, torch.matmul(A, dc))
        cc = 0.7 * F.normalize(dc, p=1, dim=0) + 0.3 * F.normalize(ev, p=1, dim=0)
        for node in self._identify_bridge_nodes(A, cc):
            cc[node] *= 1.25
        cc = F.normalize(cc, p=1, dim=0)
        self.node_importance_weights = {i: cc[i].item() for i in range(A.size(0))}
        return self.node_importance_weights

    def _identify_bridge_nodes(self, A, c):
        bridges = []
        for node in range(A.size(0)):
            nb = torch.where(A[node] > 0)[0]
            if len(nb) >= 3 and torch.var(c[nb]) > 1.5 * torch.var(c):
                bridges.append(node)
        return bridges

    def extract_metapaths(self):
        if not self.node_importance_weights:
            return []
        top10 = sorted(self.node_importance_weights.keys(),
                       key=lambda x: self.node_importance_weights[x], reverse=True)[:10]
        paths = sorted(
            [(top10[i], top10[j],
              (self.node_importance_weights[top10[i]] + self.node_importance_weights[top10[j]]) / 2)
             for i in range(len(top10)) for j in range(i+1, len(top10))],
            key=lambda x: x[2], reverse=True)
        self.metapaths = paths
        return paths

    def detect_communities(self, threshold=0.5):
        if not self._has_analyzed_structure:
            return {}
        G = nx.Graph()
        for n, w in self.node_importance_weights.items():
            G.add_node(n, importance=w)
        for s, t, w in self.metapaths:
            if w > threshold:
                G.add_edge(s, t, weight=w)
        comms = list(nx.connected_components(G))
        self.community_structure = {i: list(c) for i, c in enumerate(comms)}
        self._has_detected_communities = True
        return self.community_structure

    def extract_patterns(self):
        p = {'node_importance': self.node_importance_weights,
             'attention_weights': self.attention_weights,
             'metapaths': self.metapaths,
             'curriculum_factor': self.curriculum_factor.detach().cpu().numpy(),
             'feature_importance': self.feature_importance}
        if self._has_detected_communities:
            p['communities'] = self.community_structure
        return p