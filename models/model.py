import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax, degree
from torch_scatter import scatter_add, scatter_max, scatter_mean
import networkx as nx
from sklearn.ensemble import RandomForestClassifier


# =============================================================================
# Shared primitives
# =============================================================================

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, use_bn=True, dropout=0.0):
        super().__init__()
        h = hidden_dim or out_dim
        layers = [nn.Linear(in_dim, h),
                  nn.BatchNorm1d(h) if use_bn else nn.LayerNorm(h),
                  nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(h, out_dim),
                   nn.BatchNorm1d(out_dim) if use_bn else nn.LayerNorm(out_dim),
                   nn.GELU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim) if use_bn else nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        src, dst = edge_index
        N    = x.size(0)
        deg  = degree(edge_index[0], N).float().clamp(min=1)
        ns   = deg[src].pow(-0.5)
        nd   = deg[dst].pow(-0.5)
        xw   = self.lin(x)
        msg  = xw[src] * (ns * nd).unsqueeze(1)
        out  = scatter_add(msg, dst, dim=0, dim_size=N)
        return self.norm(F.gelu(out))


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert out_dim % num_heads == 0
        self.H = num_heads
        self.d = out_dim // num_heads
        self.W = nn.Parameter(torch.empty(num_heads, in_dim, self.d))
        self.a = nn.Parameter(torch.empty(num_heads, 2 * self.d, 1))
        self.proj = nn.Linear(out_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, edge_index):
        N, src, dst = x.size(0), edge_index[0], edge_index[1]
        heads = []
        for h in range(self.H):
            hx = x @ self.W[h]
            e  = F.leaky_relu((torch.cat([hx[src], hx[dst]], 1) @ self.a[h]), 0.2)
            a  = self.drop(softmax(e, dst, num_nodes=N))
            heads.append(scatter_add(hx[src] * a, dst, 0, dim_size=N))
        return self.proj(torch.cat(heads, 1))


def _drop_path(x, rate, training):
    if not training or rate == 0:
        return x
    keep = torch.rand(x.size(0), 1, device=x.device) > rate
    return x * keep / (1 - rate + 1e-8)

class TopologicalPositionEncoder(nn.Module):
    def __init__(self, hidden_dim, use_bn=True, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(5, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2) if use_bn else nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.LayerNorm(hidden_dim),
            nn.GELU())
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

    @torch.no_grad()
    def _compute_signatures(self, edge_index, N):
        src, dst = edge_index
        device   = edge_index.device

        # 1. degree (normalised)
        deg  = degree(src, N).float()
        deg_n = deg / (deg.max() + 1e-8)

        # 2. 2-hop degree: sum of neighbours' degrees / max
        nbr_deg   = scatter_mean(deg[src], dst, 0, dim_size=N)
        nbr_deg_n = nbr_deg / (nbr_deg.max() + 1e-8)

        # 3. local clustering coefficient:
        #    triangles_i = |{(j,k): j,k ∈ N(i), (j,k) ∈ E}| / (deg_i*(deg_i-1))
        #    Approximated efficiently: for each edge (i,j), count common neighbours
        #    using the adjacency product diagonal.
        #    We use a sparse-friendly O(E) approximation:
        #    tri_i = sum_{j in N(i)} [number of common neighbours of i and j]
        #          = (A^2)_ii  which equals sum_j A_ij * A_ji = deg for undirected
        #    Instead we compute it via dot products on indicator vectors:
        nbr_set_src = scatter_add(
            torch.ones(src.size(0), device=device), src, 0, dim_size=N)  # = deg
        # count closed triangles per node: for each edge (u,v), add deg[u]+deg[v]-2
        # then divide by deg*(deg-1) → cheap triangle approximation
        tri_raw = scatter_add((deg[src] + deg[dst] - 2).clamp(min=0),
                              src, 0, dim_size=N)
        denom   = (deg * (deg - 1)).clamp(min=1)
        clust   = (tri_raw / denom).clamp(0, 1)

        # 4. k-hop reachability (fraction of total nodes reachable in ≤2 hops)
        #    Approx: (deg + 2-hop unique reach) / N
        #    2-hop reach ≈ mean nbr degree (generous upper bound, fast)
        reach = (deg_n + nbr_deg_n).clamp(0, 1) * 0.5

        # 5. Random-walk diagonal approximation (spectral position proxy)
        #    rw_ii = 1/deg_i  (diagonal of D^{-1}A, stable and O(N))
        rw_diag = (1.0 / deg.clamp(min=1)).clamp(0, 1)

        sig = torch.stack([deg_n, nbr_deg_n, clust, reach, rw_diag], dim=1)
        return sig  # (N, 5)

    def forward(self, x, edge_index):
        N   = x.size(0)
        sig = self._compute_signatures(edge_index, N)
        pos = self.proj(sig)
        g   = self.gate(torch.cat([x, pos], 1))
        return pos, g


# =============================================================================
# Phase 1 – Engage
# Dual GCN+GAT aggregation, feature connectors, gated fusion, residual
# =============================================================================

class FeatureResourceConnector(MessagePassing):
    def __init__(self, dim, use_bn=True):
        super().__init__(aggr='mean')
        self.mlp = MLP(dim, dim, use_bn=use_bn)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return self.mlp(x_j)


class EngageModule(nn.Module):
    def __init__(self, in_node_dim, hidden_dim=128, num_heads=4,
                 dropout=0.5, use_batch_norm=True, residual=True):
        super().__init__()
        self.residual = residual
        self.node_proj  = MLP(in_node_dim, hidden_dim, use_bn=use_batch_norm)
        self.gcn        = GCNLayer(hidden_dim, hidden_dim, use_bn=use_batch_norm)
        self.gat        = MultiHeadGATLayer(hidden_dim, hidden_dim, num_heads, dropout)
        self.connector  = FeatureResourceConnector(hidden_dim, use_batch_norm)
        self.agg_w      = nn.Parameter(torch.ones(2) / 2)
        self.agg_projs  = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(2)])
        self.gate       = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.out_norm   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.LayerNorm(hidden_dim),
            nn.GELU())
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, **kw):
        fn   = self.node_proj(x)
        frn  = self.connector(fn, edge_index)
        a    = F.softmax(self.agg_w, 0)
        agg  = a[0] * self.agg_projs[0](fn) + a[1] * self.agg_projs[1](frn)
        h_gc = self.gcn(fn, edge_index)
        h_ga = self.gat(fn, edge_index)
        fus  = h_gc + h_ga
        gate = self.gate(torch.cat([agg, fus], 1))
        h    = self.out_norm(gate * agg + (1 - gate) * fus)
        if self.residual and fn.shape == h.shape:
            h = h + fn
        return self.drop(h)


# =============================================================================
# Phase 2 – Enact
# Node importance, metapath extraction, adaptive sampling,
# edge intensity, anomaly detection, feedback loop
# =============================================================================

class NodeImportanceWeighting(nn.Module):
    def __init__(self, hidden_dim, num_classes, use_bn=True):
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(num_classes, hidden_dim))
        self.weight_gen = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.LayerNorm(hidden_dim), nn.GELU())
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x, labels=None, mask=None):
        if labels is not None and mask is not None and mask.sum() > 0:
            for c in range(self.prototypes.size(0)):
                cm = (labels == c) & mask
                if cm.sum() > 0:
                    self.prototypes.data[c].mul_(0.9).add_(x[cm].mean(0).detach() * 0.1)
        sim = F.normalize(x, 2, 1) @ F.normalize(self.prototypes, 2, 1).t()
        return self.proj(torch.cat([x, x * self.weight_gen(sim)], 1))


class MetapathExtractor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, node_weights=None):
        src, dst, N = edge_index[0], edge_index[1], x.size(0)
        h1   = scatter_mean(x[src], dst, 0, dim_size=N)
        h2   = scatter_mean(h1[src], dst, 0, dim_size=N)
        w    = torch.sigmoid(node_weights).unsqueeze(1) if node_weights is not None else 0.5
        meta = w * h1 + (1 - w) * h2
        return self.proj(meta / meta.norm(2, 1, keepdim=True).clamp(min=1e-8))


class EdgeIntensityModelling(nn.Module):
    def __init__(self, hidden_dim, use_bn=True):
        super().__init__()
        self.W_e  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.LayerNorm(hidden_dim), nn.GELU())
        self.out  = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.LayerNorm(hidden_dim), nn.GELU())

    def forward(self, x, edge_index):
        src, dst = edge_index
        e_ij     = torch.sigmoid((self.W_e(x[src]) * x[dst]).sum(1, keepdim=True))
        agg      = scatter_add(self.proj(torch.cat([x[src], x[dst]], 1)) * e_ij,
                               dst, 0, dim_size=x.size(0))
        return self.out(torch.cat([x, agg], 1))


class AnomalyDetection(nn.Module):
    def __init__(self, hidden_dim, use_bn=True):
        super().__init__()
        self.ap = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.LayerNorm(hidden_dim), nn.GELU())
        self.sp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.LayerNorm(hidden_dim), nn.GELU())
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.LayerNorm(hidden_dim), nn.GELU())

    def forward(self, x, edge_index, labels=None, mask=None, num_classes=7):
        src, dst = edge_index
        nd = scatter_mean(torch.norm(x[src] - x[dst], 2, 1), src, 0, dim_size=x.size(0))
        sc = (nd - nd.min()) / (nd.max() - nd.min() + 1e-8)
        if labels is not None and mask is not None and mask.sum() > 0:
            C = min(num_classes, int(labels.max().item()) + 1)
            pr = torch.zeros(C, x.size(1), device=x.device)
            for c in range(C):
                cm = (labels == c) & mask
                if cm.sum() > 0:
                    pr[c] = x[cm].mean(0)
            la = torch.norm(x - pr[labels.clamp(0, C - 1)], 2, 1) * mask.float()
            if la.max() > 0:
                la = la / la.max()
                sc[mask] = 0.7 * la[mask] + 0.3 * sc[mask]
        h1 = scatter_mean(x[src], dst, 0, dim_size=x.size(0))
        sg = 0.5 * h1 + 0.3 * scatter_mean(h1[src], dst, 0, dim_size=x.size(0)) + 0.2 * x
        return self.proj(torch.cat([x, self.ap(x) * sc.unsqueeze(1), self.sp(sg)], 1))


class AdaptiveSampling(nn.Module):
    def __init__(self, hidden_dim, use_bn=True):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.LayerNorm(hidden_dim), nn.GELU())
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.LayerNorm(hidden_dim), nn.GELU())

    def forward(self, x, edge_index, labels=None, mask=None, num_classes=7):
        src, dst = edge_index
        ndeg = degree(src, x.size(0)).float()
        ndeg = ndeg / (ndeg.max() + 1e-8)
        if labels is None or mask is None or mask.sum() == 0:
            sim = (F.normalize(x[src], 2, 1) * F.normalize(x[dst], 2, 1)).sum(1)
            cnt = scatter_add(torch.ones_like(sim), src, 0, dim_size=x.size(0))
            sc  = 0.7 * (1 - scatter_add(sim, src, 0, dim_size=x.size(0)) / (cnt + 1e-8)) + 0.3 * ndeg
        else:
            C  = min(num_classes, int(labels.max().item()) + 1)
            pr = torch.zeros(C, x.size(1), device=x.device)
            for c in range(C):
                cm = (labels[mask] == c)
                if cm.sum() > 0:
                    pr[c] = x[mask][cm].mean(0)
            d  = torch.cdist(x, pr)
            od = d[torch.arange(x.size(0), device=x.device), labels.clamp(0, C - 1)]
            im = torch.zeros_like(d)
            im.scatter_(1, labels.clamp(0, C - 1).unsqueeze(1), float('inf'))
            sc = 0.5 * od / ((d + im).min(1).values + 1e-8) + 0.3 * ndeg + 0.2 * (1 - mask.float())
        t   = self.transform(x) * sc.unsqueeze(1)
        return self.out(0.7 * t + 0.3 * scatter_mean(t[src], dst, 0, dim_size=x.size(0)))


class EnactFeedback(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.Sigmoid())
        self.proj = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, h, ni, imb, meta):
        inp = torch.cat([h, ni * imb.unsqueeze(1), meta], 1)
        g   = self.gate(inp)
        return h * (1 - g) + self.proj(inp) * g


class EnactModule(nn.Module):
    def __init__(self, hidden_dim, num_classes, dropout=0.5, use_batch_norm=True):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_classes = num_classes

        self.node_importance = NodeImportanceWeighting(hidden_dim, num_classes, use_batch_norm)
        self.metapath        = MetapathExtractor(hidden_dim)
        self.sampling        = AdaptiveSampling(hidden_dim, use_batch_norm)
        self.edge_intensity  = EdgeIntensityModelling(hidden_dim, use_batch_norm)
        self.anomaly         = AnomalyDetection(hidden_dim, use_batch_norm)
        self.knowledge_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.knowledge_out   = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.LayerNorm(hidden_dim), nn.GELU())
        self.component_attn  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 5))
        self.feedback        = EnactFeedback(hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.LayerNorm(hidden_dim), nn.GELU())
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, partial_labels=None, mask=None, **kw):
        ni   = self.node_importance(x, partial_labels, mask)
        ni_w = torch.sigmoid(ni.mean(1))
        meta = self.metapath(x, edge_index, ni_w)
        samp = self.sampling(x, edge_index, partial_labels, mask, self.num_classes)
        ei   = self.edge_intensity(x, edge_index)
        anom = self.anomaly(x, edge_index, partial_labels, mask, self.num_classes)
        kp   = self.knowledge_proj(self.node_importance.prototypes.detach())
        sim  = F.normalize(x, 2, 1) @ F.normalize(kp, 2, 1).t()
        kf   = self.knowledge_out(torch.cat([x, sim @ kp], 1))
        aw   = F.softmax(self.component_attn(x), 1)
        h    = sum(aw[:, i:i+1] * c for i, c in enumerate([ni, samp, ei, anom, kf]))
        imb  = self._imbalance(x, partial_labels, mask)
        h    = self.feedback(h, ni, imb, meta)
        return self.drop(self.out(h))

    def _imbalance(self, x, labels, mask):
        sc = torch.ones(x.size(0), device=x.device)
        if labels is not None and mask is not None and mask.sum() > 0:
            C   = min(self.num_classes, int(labels.max().item()) + 1)
            cnt = torch.zeros(C, device=x.device)
            for c in range(C):
                cnt[c] = ((labels == c) & mask).float().sum()
            if cnt.sum() > 0:
                sc[mask] = (1 - cnt / cnt.sum())[labels[mask].clamp(0, C - 1)]
        return sc


# =============================================================================
# Phase 3 – Embed
# Memory-augmented update, knowledge graph integration,
# gradient-historical updates (4 strategies), reflective learning
# =============================================================================

class MemoryAugmentedUpdate(nn.Module):
    def __init__(self, hidden_dim, num_slots=16, use_bn=True):
        super().__init__()
        self.M_keys = nn.Parameter(torch.empty(num_slots, hidden_dim))
        self.M_vals = nn.Parameter(torch.empty(num_slots, hidden_dim))
        self.ctrl   = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.LayerNorm(hidden_dim), nn.Sigmoid())
        self.proc   = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.LayerNorm(hidden_dim), nn.GELU())
        nn.init.xavier_uniform_(self.M_keys)
        nn.init.xavier_uniform_(self.M_vals)

    def forward(self, x):
        att = F.softmax(F.normalize(x @ self.M_keys.t(), dim=1), 1)
        r   = att @ self.M_vals
        with torch.no_grad():
            self.M_vals.data.mul_(0.95).add_(0.05 * F.normalize(att.t(), 1, 1) @ x)
        c = self.ctrl(torch.cat([x, r], 1))
        return c * self.proc(torch.cat([x, r], 1)) + (1 - c) * x


class KnowledgeGraphIntegration(nn.Module):
    def __init__(self, hidden_dim, use_bn=True):
        super().__init__()
        self.ext  = MLP(hidden_dim, hidden_dim, use_bn=use_bn)
        self.inte = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.LayerNorm(hidden_dim), nn.GELU())

    def forward(self, x, edge_index):
        src, dst, N = edge_index[0], edge_index[1], x.size(0)
        k  = self.ext(x)
        h1 = scatter_mean(k[src], dst, 0, dim_size=N)
        h2 = scatter_mean(h1[src], dst, 0, dim_size=N)
        return self.inte(torch.cat([x, h1, h2], 1))


class GradientHistoricalUpdate(nn.Module):
    def __init__(self, hidden_dim, num_classes, use_bn=True):
        super().__init__()
        self.W_transfer  = nn.Linear(hidden_dim, hidden_dim)
        self.value_net   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2) if use_bn else nn.LayerNorm(hidden_dim // 2),
            nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.enhance     = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.strategy_sel = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2) if use_bn else nn.LayerNorm(hidden_dim // 2),
            nn.GELU(), nn.Linear(hidden_dim // 2, 3), nn.Softmax(dim=1))
        self.strategies   = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, use_bn=use_bn),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2) if use_bn else nn.LayerNorm(hidden_dim // 2),
                nn.GELU(), nn.Linear(hidden_dim // 2, hidden_dim)),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2) if use_bn else nn.LayerNorm(hidden_dim * 2),
                nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))])
        self.integrate = MLP(hidden_dim, hidden_dim, use_bn=use_bn)

    def forward(self, x, rewards):
        h_t  = self.W_transfer(x)
        adv  = rewards.unsqueeze(1) - self.value_net(x)
        h_h  = x + self.enhance(torch.cat([x, adv], 1))
        h_c  = x * self.context_gate(torch.cat([h_t, h_h], 1))
        base = h_t + h_h + h_c
        sw   = self.strategy_sel(base)
        return self.integrate(sum(self.strategies[i](base) * sw[:, i:i+1] for i in range(3)))


class ReflectiveLearning(nn.Module):
    def __init__(self, hidden_dim, n=4, use_bn=True):
        super().__init__()
        self.reflect = nn.Sequential(
            nn.Linear(hidden_dim * (n + 1), hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2) if use_bn else nn.LayerNorm(hidden_dim * 2),
            nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2) if use_bn else nn.LayerNorm(hidden_dim // 2),
            nn.GELU(), nn.Linear(hidden_dim // 2, hidden_dim), nn.Sigmoid())

    def forward(self, x, components):
        r = self.reflect(torch.cat([x] + components, 1))
        g = self.gate(torch.cat([x, r], 1))
        return g * r + (1 - g) * x


class EmbedModule(nn.Module):
    def __init__(self, hidden_dim, num_classes, dropout=0.5, use_batch_norm=True):
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
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.LayerNorm(hidden_dim), nn.GELU())
        self.cls  = nn.Linear(hidden_dim, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, partial_labels=None, mask=None, **kw):
        kg   = self.knowledge(x, edge_index)
        mem  = self.memory(x)
        rwd  = self._rewards(x, edge_index, partial_labels, mask)
        gh   = self.grad_hist(x, rwd)
        comp = [kg, mem, gh, kg]
        aw   = F.softmax(self.attn(x), 1)
        h    = sum(aw[:, i:i+1] * c for i, c in enumerate(comp))
        h    = h + 0.3 * self.reflective(x, comp)
        emb  = self.drop(self.out(h))
        return emb, self.cls(emb)

    def _rewards(self, x, edge_index, labels, mask):
        N, r = x.size(0), torch.zeros(x.size(0), device=x.device)
        if labels is not None and mask is not None and mask.sum() > 0:
            with torch.no_grad():
                r[(self.cls(x).argmax(1) == labels) & mask] = 1.0
            src, dst = edge_index
            for _ in range(2):
                prop, _ = scatter_max(0.7 * r[src], dst, 0, dim_size=N)
                r = torch.max(r, prop)
        else:
            d = degree(edge_index[0], N).float()
            r = d / (d.max() + 1e-8)
        return r


# =============================================================================
# Residual GCN+GAT block with stochastic depth (used in backbone)
# =============================================================================

class ResidualGNNBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.5,
                 use_bn=True, drop_path_rate=0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.gcn  = GCNConv(hidden_dim, hidden_dim)
        self.gat  = GATConv(hidden_dim, hidden_dim // num_heads,
                            heads=num_heads, dropout=dropout, concat=True)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.LayerNorm(hidden_dim),
            nn.GELU())
        self.skip          = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.drop          = nn.Dropout(dropout)
        self.drop_path_rate = drop_path_rate

    def forward(self, x, edge_index):
        h = self.drop(self.fuse(torch.cat([F.gelu(self.gcn(x, edge_index)),
                                           F.gelu(self.gat(x, edge_index))], 1)))
        return _drop_path(h, self.drop_path_rate, self.training) + self.skip(x)


class CL3EGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes,
                 backbone_layers=3,
                 num_heads=4,
                 dropout=0.5,
                 use_batch_norm=True,
                 aux_weight=0.15,
                 label_smoothing=0.15,
                 attention_type='sigmoid'):
        super().__init__()
        self.hidden_dim      = hidden_dim
        self.num_classes     = num_classes
        self.aux_weight      = aux_weight
        self.label_smoothing = label_smoothing
        self.num_gcn_layers  = backbone_layers
        self.attention_type  = attention_type
        self.dropout_rate    = dropout

        nh = max(1, hidden_dim // 32)

        # Topological Position Encoder — structural bias for all three phases
        self.tpe = TopologicalPositionEncoder(hidden_dim, use_bn=use_batch_norm,
                                              dropout=dropout)

        self.engage = EngageModule(in_dim, hidden_dim, num_heads=nh,
                                   dropout=dropout, use_batch_norm=use_batch_norm)
        self.enact  = EnactModule(hidden_dim, num_classes,
                                  dropout=dropout, use_batch_norm=use_batch_norm)
        self.embed  = EmbedModule(hidden_dim, num_classes,
                                  dropout=dropout, use_batch_norm=use_batch_norm)

        # optional residual backbone (V1/V2 mode)
        self.use_backbone = backbone_layers > 0
        if self.use_backbone:
            dp = [x.item() for x in torch.linspace(0.0, 0.2, backbone_layers)]
            self.blocks  = nn.ModuleList([
                ResidualGNNBlock(hidden_dim, num_heads=nh, dropout=dropout,
                                 use_bn=use_batch_norm, drop_path_rate=dp[i])
                for i in range(backbone_layers)])
            self.jk_proj = nn.Sequential(
                nn.Linear(hidden_dim * (backbone_layers + 1), hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.LayerNorm(hidden_dim),
                nn.GELU())
            # three-stage attention for V1/V2-style representation aggregation
            if attention_type == 'sigmoid':
                self.s1 = nn.Linear(hidden_dim, 1)
                self.s2 = nn.Linear(hidden_dim, 1)
                self.s3 = nn.Linear(hidden_dim, 1)
            else:
                self.s1 = nn.Linear(hidden_dim, hidden_dim)
                self.s2 = nn.Linear(hidden_dim, hidden_dim)
                self.s3 = nn.Linear(hidden_dim, hidden_dim)

        # phase-to-phase gated skip connections
        self.gate_e2n = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.LayerNorm(hidden_dim),
            nn.Sigmoid())
        self.gate_n2e = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.LayerNorm(hidden_dim),
            nn.Sigmoid())

        # auxiliary classification heads (intermediate supervision)
        _aux = lambda: nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes))
        self.aux_engage = _aux()
        self.aux_enact  = _aux()

        # learnable ensemble weights over the three phase heads
        self.phase_weights = nn.Parameter(torch.tensor([0.2, 0.3, 0.5]))

        # curriculum factor (V1/V2 style per-class scaling)
        self.curriculum_factor = nn.Parameter(torch.ones(num_classes))

        # V2-style structural analysis state
        self.node_importance_weights   = {}
        self.attention_weights         = {}
        self.metapaths                 = []
        self.community_structure       = {}
        self._has_analyzed_structure   = False
        self._has_detected_communities = False
        self.feature_importance        = {}

    def forward(self, x, edge_index, edge_attr=None,
                partial_labels=None, mask=None, **kw):

        # Topological position encoding — structural signature injected into all phases
        # Compute TPE after Engage so gate operates in hidden_dim space
        h_e = self.engage(x, edge_index)
        tpe_bias, tpe_gate = self.tpe(h_e, edge_index)
        h_e = h_e + tpe_gate * tpe_bias

        # optional backbone between Engage and Enact
        if self.use_backbone:
            layer_outs = [h_e]
            h_b = h_e
            for block in self.blocks:
                h_b = block(h_b, edge_index)
                layer_outs.append(h_b)
            h_b = self.jk_proj(torch.cat(layer_outs, 1))

            mid = len(layer_outs) // 2
            r1  = (layer_outs[0]   * torch.sigmoid(self.s1(layer_outs[0]))).sum(0)
            r2  = (layer_outs[mid] * torch.sigmoid(self.s2(layer_outs[mid]))).sum(0)
            r3  = (h_b             * torch.sigmoid(self.s3(h_b))).sum(0)
            combined_repr = r1 + r2 + r3

            s3w = torch.sigmoid(self.s3(h_b))
            for i in range(h_b.size(0)):
                self.attention_weights[i] = s3w[i].item()

            # blend backbone output with engage via gated skip
            g    = self.gate_e2n(h_b)
            h_en_in = h_b * g
        else:
            combined_repr = None
            g        = self.gate_e2n(h_e)
            h_en_in  = h_e * g

        # Phase 2: Enact  (+TPE bias)
        h_n = self.enact(h_en_in + tpe_bias, edge_index,
                         partial_labels=partial_labels, mask=mask)
        h_n = h_n + h_e   # residual from engage

        # Phase 3: Embed  (+TPE bias)
        blend      = self.gate_n2e(torch.cat([h_n, h_e], 1))
        h_embed_in = blend * h_n + (1 - blend) * h_e + tpe_bias
        emb, logits_embed = self.embed(h_embed_in, edge_index,
                                       partial_labels=partial_labels, mask=mask)

        # auxiliary heads
        logits_e = self.aux_engage(h_e)
        logits_n = self.aux_enact(h_n)

        # learned ensemble — no curriculum_factor here; applied in loss only
        pw     = F.softmax(self.phase_weights, 0)
        logits = pw[0] * logits_e + pw[1] * logits_n + pw[2] * logits_embed

        return emb, logits, logits_e, logits_n

    def aux_loss(self, lg_e, lg_n, targets):
        def _sce(logits, tgt):
            nc   = logits.size(1)
            s    = self.label_smoothing
            soft = torch.zeros_like(logits).scatter_(1, tgt.unsqueeze(1), 1.0)
            soft = soft * (1 - s) + s / nc
            return -(soft * F.log_softmax(logits, 1)).sum(1).mean()
        return self.aux_weight * 0.5 * (_sce(lg_e, targets) + _sce(lg_n, targets))

    def curriculum_loss(self, outputs, targets):
        loss = F.cross_entropy(outputs, targets, reduction='none')
        return (loss * self.curriculum_factor[targets]).mean()

    def adjust_curriculum(self, difficulty_scores):
        with torch.no_grad():
            for c, d in difficulty_scores.items():
                if c < self.num_classes:
                    self.curriculum_factor[c] = float(2.0 - d)
        return self.curriculum_factor.detach().cpu().numpy()

    # ── V2-style structural analysis (preserved) ──────────────────────────────

    def facilitate_learning(self, adj_matrices):
        self.calculate_node_importance(adj_matrices)
        self.extract_metapaths()
        self._has_analyzed_structure = True
        top5 = sorted(self.node_importance_weights.items(),
                      key=lambda x: x[1], reverse=True)[:5]
        return {'top_nodes': top5, 'metapaths': self.metapaths[:5],
                'attention_weights': dict(sorted(self.attention_weights.items(),
                                                  key=lambda x: x[1], reverse=True)[:5]),
                'structure_analyzed': True}

    def calculate_node_importance(self, adj_matrices):
        if not adj_matrices:
            return {}
        A  = adj_matrices[0]
        dc = A.sum(1)
        ev = A @ (A @ dc)
        cc = 0.7 * F.normalize(dc, 1, 0) + 0.3 * F.normalize(ev, 1, 0)
        for node in self._identify_bridge_nodes(A, cc):
            cc[node] *= 1.25
        cc = F.normalize(cc, 1, 0)
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
        top10 = sorted(self.node_importance_weights, key=self.node_importance_weights.get,
                       reverse=True)[:10]
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
        self.community_structure       = {i: list(c) for i, c in enumerate(nx.connected_components(G))}
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


# =============================================================================
# CL3E_ModelV2_Original — standalone original V2 architecture
#
# Preserved exactly as designed: Engage → Enact → GCN backbone →
# three-stage sigmoid attention → final_gcn classifier.
# Includes all V2-specific capabilities: community detection, metapath
# extraction, node importance weighting, bridge node identification,
# curriculum factor, and pattern extraction.
#
# Use this when you need the original V2 behaviour independently of CL3EGNN.
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

        self.engage = EngageModule(in_node_dim=in_dim, hidden_dim=hidden_dim,
                                   dropout=dropout, use_batch_norm=use_batch_norm)
        self.enact  = EnactModule(hidden_dim=hidden_dim, num_classes=num_classes,
                                  dropout=dropout, use_batch_norm=use_batch_norm)

        self.gcn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.final_gcn  = GCNConv(hidden_dim, num_classes)
        self.dropout    = nn.Dropout(dropout)
        self.curriculum_factor = nn.Parameter(torch.ones(num_classes))

        if attention_type == 'sigmoid':
            self.s1 = nn.Linear(hidden_dim, 1)
            self.s2 = nn.Linear(hidden_dim, 1)
            self.s3 = nn.Linear(hidden_dim, 1)
        else:
            self.s1 = nn.Linear(hidden_dim, hidden_dim)
            self.s2 = nn.Linear(hidden_dim, hidden_dim)
            self.s3 = nn.Linear(hidden_dim, hidden_dim)

        self.node_importance_weights    = {}
        self.attention_weights          = {}
        self.metapaths                  = []
        self.community_structure        = {}
        self._has_analyzed_structure    = False
        self._has_detected_communities  = False
        self.knowledge_cache            = {}
        self.rewards                    = {}
        self.reward_history             = {}
        self.performance_metrics        = {}
        self.transfer_history           = []
        self.adaptation_history         = []
        self.feature_importance         = {}

    def forward(self, x, edge_index, partial_labels=None, mask=None, **kw):
        h_e = self.engage(x, edge_index)
        h_n = self.enact(h_e, edge_index, partial_labels=partial_labels, mask=mask)
        h   = h_n + h_e

        intermediates = [h]
        for layer in self.gcn_layers:
            h = self.dropout(F.relu(layer(h, edge_index)))
            intermediates.append(h)

        mid      = len(intermediates) // 2
        r1       = torch.sum(intermediates[0]   * torch.sigmoid(self.s1(intermediates[0])),  dim=0)
        r2       = torch.sum(intermediates[mid]  * torch.sigmoid(self.s2(intermediates[mid])), dim=0)
        r3       = torch.sum(h                   * torch.sigmoid(self.s3(h)),                  dim=0)
        combined = r1 + r2 + r3

        s3w = torch.sigmoid(self.s3(h))
        for i in range(h.size(0)):
            self.attention_weights[i] = s3w[i].item()

        out = self.final_gcn(h, edge_index)
        out = out * self.curriculum_factor.unsqueeze(0)
        return out, combined

    def adjust_curriculum(self, difficulty_scores):
        with torch.no_grad():
            for c, d in difficulty_scores.items():
                if c < self.num_classes:
                    self.curriculum_factor[c] = float(2.0 - d)
        return self.curriculum_factor.detach().cpu().numpy()

    def facilitate_learning(self, adj_matrices):
        self.calculate_node_importance(adj_matrices)
        self.extract_metapaths()
        self._has_analyzed_structure = True
        top5 = sorted(self.node_importance_weights.items(),
                      key=lambda x: x[1], reverse=True)[:5]
        return {'top_nodes': top5,
                'metapaths': self.metapaths[:5],
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
        top10 = sorted(self.node_importance_weights,
                       key=self.node_importance_weights.get, reverse=True)[:10]
        paths = sorted(
            [(top10[i], top10[j],
              (self.node_importance_weights[top10[i]] + self.node_importance_weights[top10[j]]) / 2)
             for i in range(len(top10)) for j in range(i + 1, len(top10))],
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
        self.community_structure        = {i: list(c)
                                           for i, c in enumerate(nx.connected_components(G))}
        self._has_detected_communities  = True
        return self.community_structure

    def extract_patterns(self):
        p = {'node_importance':  self.node_importance_weights,
             'attention_weights': self.attention_weights,
             'metapaths':         self.metapaths,
             'curriculum_factor': self.curriculum_factor.detach().cpu().numpy(),
             'feature_importance': self.feature_importance}
        if self._has_detected_communities:
            p['communities'] = self.community_structure
        return p

def CL3EClassifier(in_dim, hidden_dim, num_classes, dropout=0.5,
                   use_batch_norm=True, **kw):
    return CL3EGNN(in_dim, hidden_dim, num_classes,
                   backbone_layers=0, dropout=dropout,
                   use_batch_norm=use_batch_norm, **kw)


def CL3E_ModelV1(in_dim, hidden_dim, num_classes, num_layers=3,
                 dropout=0.5, attention_type='sigmoid',
                 use_batch_norm=True, **kw):
    return CL3EGNN(in_dim, hidden_dim, num_classes,
                   backbone_layers=num_layers, dropout=dropout,
                   attention_type=attention_type,
                   use_batch_norm=use_batch_norm, **kw)


def CL3E_ModelV2(in_dim, hidden_dim, num_classes,
                 graph_data=None, num_nodes=None, model=None,
                 num_layers=2, dropout=0.5,
                 attention_type='sigmoid', use_batch_norm=True,
                 max_metapaths=30, **kw):
    """Returns the original standalone V2 architecture (CL3E_ModelV2_Original)."""
    return CL3E_ModelV2(
        in_dim=in_dim, hidden_dim=hidden_dim, num_classes=num_classes,
        graph_data=graph_data, num_nodes=num_nodes, model=model,
        num_layers=num_layers, dropout=dropout,
        attention_type=attention_type, use_batch_norm=use_batch_norm,
        max_metapaths=max_metapaths)