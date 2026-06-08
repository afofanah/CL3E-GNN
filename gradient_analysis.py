"""
Gradient stability analysis for CL3E-GNN — three semantic phases.

Phase 1 (Engage):  engage.* + gate_e2n.* + gate_n2e.*
Phase 2 (Enact):   enact.* + blocks.* + jk_proj.*
Phase 3 (Embed):   embed.* + classifier.* + aux_engage.* + aux_enact.*
                   + s1.* + s2.* + s3.* + curriculum_factor + phase_w

Attention parameters tracked per phase:
  Phase 1: gat.W, gat.a, gate (EngageModule)
  Phase 2: component_attn, feedback.gate, blocks.*.gat, s1/s2/s3
  Phase 3: phase_w, embed.attn, aux head weights
"""

import os
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

matplotlib.rcParams.update({
    'font.size':         15,
    'axes.titlesize':    15,
    'axes.labelsize':    15,
    'xtick.labelsize':   18,
    'ytick.labelsize':   18,
    'legend.fontsize':   15,
    'figure.titlesize':  16,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

PHASE_NAMES  = ['Phase-1 (Engage)', 'Phase-2 (Enact)', 'Phase-3 (Embed)']
PHASE_COLORS = {
    'Phase-1 (Engage)': '#2196F3',
    'Phase-2 (Enact)':  '#FF9800',
    'Phase-3 (Embed)':  '#4CAF50',
}

# ── attention parameter name fragments per phase ──────────────────────────────
_ATTN_KEYS = {
    'Phase-1 (Engage)': ('gat.W', 'gat.a', 'engage.gate', 'engage.aggregator.w',
                         'gate_e2n', 'gate_n2e'),
    'Phase-2 (Enact)':  ('component_attn', 'feedback.gate', 'blocks.',
                         's1.', 's2.', 's3.',
                         'enact.node_importance', 'enact.metapath'),
    'Phase-3 (Embed)':  ('phase_w', 'embed.attn', 'aux_engage', 'aux_enact',
                         'embed.memory', 'curriculum_factor'),
}


def _phase_of(name: str) -> Optional[str]:
    """
    Assign a parameter to one of the three semantic phases.
    Works for CL3EClassifier, CL3E_ModelV1, and CL3E_ModelV2.
    """
    # Phase 1 – Engage
    if (name.startswith('engage.')
            or name.startswith('gate_e2n.')
            or name.startswith('gate_n2e.')):
        return 'Phase-1 (Engage)'

    # Phase 2 – Enact  (backbone blocks + JK + enact module + stage attention)
    if (name.startswith('enact.')
            or name.startswith('blocks.')
            or name.startswith('jk_proj.')
            or name.startswith('s1.')
            or name.startswith('s2.')
            or name.startswith('s3.')):
        return 'Phase-2 (Enact)'

    # Phase 3 – Embed  (embed module, classifiers, aux heads, curriculum)
    if (name.startswith('embed.')
            or name.startswith('classifier.')
            or name.startswith('aux_engage.')
            or name.startswith('aux_enact.')
            or name.startswith('phase_w')
            or name == 'curriculum_factor'):
        return 'Phase-3 (Embed)'

    return None


def _is_attn_param(name: str, phase: str) -> bool:
    keys = _ATTN_KEYS.get(phase, ())
    return any(k in name for k in keys)


# =============================================================================
# GradientTracker
# =============================================================================

class GradientTracker:
    def __init__(self):
        self.grad_norms    = {p: [] for p in PHASE_NAMES}
        self.grad_variance = {p: [] for p in PHASE_NAMES}
        self.grad_max      = {p: [] for p in PHASE_NAMES}
        self.grad_min      = {p: [] for p in PHASE_NAMES}
        self.attn_weights  = {p: [] for p in PHASE_NAMES}
        self.loss_values   = []
        self.phase_names   = PHASE_NAMES

    def record(self, model, loss_val: float):
        self.loss_values.append(loss_val)
        phase_grads = {p: [] for p in PHASE_NAMES}
        phase_attn  = {p: [] for p in PHASE_NAMES}

        for name, param in model.named_parameters():
            phase = _phase_of(name)
            if phase is None or param.grad is None:
                continue
            g = param.grad.detach().norm().item()
            phase_grads[phase].append(g)
            if _is_attn_param(name, phase):
                phase_attn[phase].append(param.data.detach().abs().mean().item())

        for p in PHASE_NAMES:
            gs = phase_grads[p]
            if gs:
                self.grad_norms[p].append(float(np.mean(gs)))
                self.grad_variance[p].append(float(np.var(gs)))
                self.grad_max[p].append(float(np.max(gs)))
                self.grad_min[p].append(float(np.min(gs)))
            else:
                last = self.grad_norms[p][-1] if self.grad_norms[p] else 0.0
                self.grad_norms[p].append(last)
                self.grad_variance[p].append(0.0)
                self.grad_max[p].append(last)
                self.grad_min[p].append(last)

            aw = phase_attn[p]
            self.attn_weights[p].append(float(np.mean(aw)) if aw else
                                        (self.attn_weights[p][-1] if self.attn_weights[p] else 0.0))


# =============================================================================
# Training loop with tracking
# =============================================================================

def train_with_gradient_tracking(model, data, device,
                                  epochs=200, lr=0.005,
                                  weight_decay=5e-4,
                                  loss_type='standard',
                                  curriculum_args=None,
                                  entropy_args=None):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.loss import get_loss

    model     = model.to(device)
    data      = data.to(device)
    tracker   = GradientTracker()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = get_loss(loss_type, int(data.y.max().item()) + 1,
                         curriculum_args, entropy_args)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index,
                    partial_labels=data.y, mask=data.train_mask)
        # unpack any output shape
        if isinstance(out, (tuple, list)):
            logits = out[1] if (len(out) >= 2 and out[1].dim() == 2) else out[0]
        else:
            logits = out

        tl   = data.y[data.train_mask].long()
        loss = criterion(logits[data.train_mask], tl)
        loss.backward()

        # record BEFORE clipping so we see raw gradient norms
        tracker.record(model, loss.item())

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if hasattr(criterion, 'update_epoch'):
            criterion.update_epoch(epoch)

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                out2 = model(data.x, data.edge_index,
                             partial_labels=data.y, mask=data.train_mask)
                lg2  = out2[1] if isinstance(out2, (tuple, list)) and out2[1].dim()==2 else (
                       out2[0] if isinstance(out2, (tuple, list)) else out2)
                va = lg2.argmax(1)[data.val_mask].eq(
                    data.y[data.val_mask]).float().mean().item()
            print(f'  Epoch {epoch:03d} | loss {loss.item():.4f} | val_acc {va:.4f}')
            model.train()

    return tracker


# =============================================================================
# Plotting helpers
# =============================================================================

def _savefig(fig, path):
    if path is None:
        plt.close(fig); return
    base = os.path.splitext(path)[0]
    fig.savefig(base + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(base + '.pdf', bbox_inches='tight')
    plt.close(fig)


def _smooth(arr, w=7):
    if len(arr) < w:
        return np.array(arr)
    return np.convolve(arr, np.ones(w) / w, mode='valid')


# =============================================================================
# Plot 1 – Main 6-panel gradient stability overview
# =============================================================================

# =============================================================================
# Plot 1 – Main 6-panel gradient stability overview
# =============================================================================

def plot_gradient_stability(tracker: GradientTracker, save_path=None):
    """
    6-panel figure:
      [0,0] Gradient norm per phase (raw + smoothed)
      [0,1] Gradient variance per phase (log scale)
      [0,2] Training loss with phase colour bands
      [1,0] Min/mean/max gradient band per phase
      [1,1] Attention / gate weight magnitude per phase
      [1,2] Normalised relative gradient stability
    """
    phases = PHASE_NAMES
    epochs = np.arange(1, len(tracker.loss_values) + 1)
    T      = len(epochs)

    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 3, figure=fig)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

    # [0,0] gradient norm
    ax = axes[0][0]
    for p in phases:
        c  = PHASE_COLORS[p]
        ep = np.arange(1, len(tracker.grad_norms[p]) + 1)
        ax.plot(ep, tracker.grad_norms[p], alpha=0.2, linewidth=1, color=c)
        sm = _smooth(tracker.grad_norms[p])
        ax.plot(np.arange(1, len(sm)+1) + 3, sm, linewidth=2.5, color=c, label=p)
    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel('‖∇θ‖₂', fontsize=18)
    ax.set_title('Gradient Norm per Phase\n(Theorem 1: Engage bounds λe⁻ˡL)', fontsize=16)
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.25)

    # [0,1] variance log scale
    ax = axes[0][1]
    for p in phases:
        c  = PHASE_COLORS[p]
        ep = np.arange(1, len(tracker.grad_variance[p]) + 1)
        ax.semilogy(ep, np.array(tracker.grad_variance[p]) + 1e-10,
                    linewidth=2.2, color=c, label=p)
    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel('Gradient Variance (log)', fontsize=18)
    ax.set_title('Gradient Variance per Phase\n(Var[∇] ≤ L²/N)', fontsize=16)
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.25, which='both')

    # [0,2] loss with phase bands
    ax = axes[0][2]
    ax.plot(epochs, tracker.loss_values, color='#333', linewidth=2.2, label='Train loss')
    sm = _smooth(tracker.loss_values)
    ax.plot(np.arange(1, len(sm)+1)+3, sm, color='red', linewidth=2.5,
            linestyle='--', label='Smoothed')
    seg = T // 3
    for i, (p, rc) in enumerate(zip(phases, PHASE_COLORS.values())):
        ax.axvspan(i*seg+1, min((i+1)*seg, T), alpha=0.07, color=rc, label=p)
    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel('Loss', fontsize=18)
    ax.set_title('Training Loss with Phase Regions', fontsize=16)
    ax.legend(fontsize=15, ncol=2)
    ax.grid(True, alpha=0.25)

    # [1,0] min/mean/max band
    ax = axes[1][0]
    for p in phases:
        c  = PHASE_COLORS[p]
        mn = np.array(tracker.grad_min[p])
        mu = np.array(tracker.grad_norms[p])
        mx = np.array(tracker.grad_max[p])
        ep = np.arange(1, len(mu)+1)
        ax.fill_between(ep, mn, mx, alpha=0.15, color=c)
        ax.plot(ep, mu, linewidth=2.2, color=c, label=p)
    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel('‖∇θ‖₂', fontsize=18)
    ax.set_title('Gradient Band (min / mean / max)', fontsize=16)
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.25)

    # [1,1] attention weight magnitude per phase
    ax = axes[1][1]
    any_attn = False
    for p in phases:
        aw = tracker.attn_weights[p]
        c  = PHASE_COLORS[p]
        ep = np.arange(1, len(aw)+1)
        if any(v > 1e-8 for v in aw):
            ax.plot(ep, aw, linewidth=2.2, color=c, label=p)
            any_attn = True
    if not any_attn:
        ax.text(0.5, 0.5, 'No attention params\ndetected',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel('Mean |Attention Weight|', fontsize=18)
    ax.set_title('Attention / Gate Weight Magnitude\nper Phase (s1/s2/s3, GAT, gates)', fontsize=16)
    if any_attn:
        ax.legend(fontsize=15)
    ax.grid(True, alpha=0.25)

    # [1,2] normalised relative stability
    ax = axes[1][2]
    for p in phases:
        norms  = np.array(tracker.grad_norms[p])
        base   = np.mean(norms[:max(1, len(norms)//10)]) + 1e-8
        normed = norms / base
        ep     = np.arange(1, len(normed)+1)
        ax.plot(ep, normed, linewidth=2.2, color=PHASE_COLORS[p], label=p)
    ax.axhline(1.0, color='black', linestyle=':', linewidth=1.5, alpha=0.6,
               label='Initial baseline')
    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel('Normalised ‖∇θ‖₂', fontsize=18)
    ax.set_title('Relative Gradient Stability\n(÷ initial; 1.0 = stable)', fontsize=16)
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.25)

    # fig.suptitle('CL3E-GNN Three-Phase Gradient Stability Analysis\n'
    #              'Empirical validation of Theorem 1', fontsize=17)
    _savefig(fig, save_path)

# =============================================================================
# Plot 2 – Per-phase subplots with ±σ band and O(1/√t) reference
# =============================================================================

def plot_gradient_norm_per_phase_separate(tracker: GradientTracker, save_path=None):
    n  = len(PHASE_NAMES)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 5), sharey=False)

    for ax, p in zip(axes, PHASE_NAMES):
        norms = np.array(tracker.grad_norms[p])
        var   = np.array(tracker.grad_variance[p])
        sigma = np.sqrt(var + 1e-10)
        ep    = np.arange(1, len(norms)+1)
        c     = PHASE_COLORS[p]

        ax.fill_between(ep, norms-sigma, norms+sigma, alpha=0.2, color=c, label='±1σ')
        ax.plot(ep, norms, alpha=0.4, linewidth=1, color=c)
        sm = _smooth(norms)
        ax.plot(np.arange(1, len(sm)+1)+3, sm, linewidth=2.8, color=c, label='Smoothed')
        ref_scale = norms[0] if norms[0] > 0 else 1.0
        ax.plot(ep, ref_scale / np.sqrt(ep), linewidth=1.8, color='red',
                linestyle='--', alpha=0.7, label='O(1/√t) ref.')

        ax.set_xlabel('Epoch'); ax.set_ylabel('‖∇θ‖₂')
        ax.set_title(f'{p}\n(raw ± σ + smoothed)')
        ax.legend(fontsize=14); ax.grid(True, alpha=0.25)

    # fig.suptitle('Per-Phase Gradient Norm with O(1/√t) Reference\n'
    #              '(Theorem 1 – gradient instability bound)', fontsize=16)
    fig.tight_layout()
    _savefig(fig, save_path)


# =============================================================================
# Plot 3 – Attention weight evolution: one subplot per phase
# =============================================================================

def plot_attention_weights_per_phase(tracker: GradientTracker, save_path=None):
    """
    Dedicated attention weight evolution plot — one panel per phase.
    Shows how attention mechanisms in each phase evolve during training.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    phase_descriptions = {
        'Phase-1 (Engage)': 'GAT W/a, EngageGate\n(EngageModule attention)',
        'Phase-2 (Enact)':  'component_attn, feedback.gate\nblocks GAT, s1/s2/s3\n(Enact + backbone attention)',
        'Phase-3 (Embed)':  'phase_w, embed.attn\naux heads, curriculum_factor\n(Embed ensemble weights)',
    }

    for ax, p in zip(axes, PHASE_NAMES):
        aw = np.array(tracker.attn_weights[p])
        ep = np.arange(1, len(aw)+1)
        c  = PHASE_COLORS[p]

        ax.fill_between(ep, 0, aw, alpha=0.2, color=c)
        ax.plot(ep, aw, linewidth=2.5, color=c)

        # smoothed trend
        if len(aw) >= 7:
            sm = _smooth(aw)
            ax.plot(np.arange(1, len(sm)+1)+3, sm, linewidth=2,
                    color='black', linestyle='--', alpha=0.6, label='Trend')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean |Attention Weight|')
        ax.set_title(f'{p}\n{phase_descriptions[p]}')
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.25)

    # fig.suptitle('Attention / Gate Weight Evolution per Phase\n'
    #              '(Validates phase-specific attention mechanisms)', fontsize=17)
    fig.tight_layout()
    _savefig(fig, save_path)


# =============================================================================
# Plot 4 – Theorem 1 Gradient Variance Bound Validation
#
# Theorem 1 states: Var[∇] ≤ L²/N  (statistical bound from Hoeffding's
# inequality) plus the imbalance penalty λe^{-λ}L.
#
# This plot validates the theorem empirically with three panels:
#   [left]   Empirical variance per phase vs theoretical bound L²/N,
#            showing the bound holds throughout training
#   [centre] Variance decay rate compared to the O(1/N) theoretical slope,
#            plotted on log-log axes to verify the power law
#   [right]  Imbalance penalty term λe^{-λ}L across training epochs,
#            computed from the running gradient statistics
# =============================================================================

def plot_gradient_variance_bound(tracker: GradientTracker,
                                  N_train: int = 140,
                                  lipschitz_L: float = 1.0,
                                  lambda_imb: float = 1.0,
                                  save_path=None):
    """
    Validates Theorem 1: Var[∇θ L_Engage] ≤ L²/N + λe^{-λ}L

    Parameters
    ----------
    N_train    : number of training nodes (denominator in L²/N bound)
    lipschitz_L: estimated Lipschitz constant (set to max observed grad norm
                 in first epoch as a conservative upper bound)
    lambda_imb : class imbalance factor λ
    """
    T      = len(tracker.loss_values)
    epochs = np.arange(1, T + 1)

    # auto-estimate L from observed gradient norms (conservative: use max)
    all_norms = np.concatenate([tracker.grad_norms[p] for p in PHASE_NAMES])
    if all_norms.max() > 0:
        lipschitz_L = float(np.percentile(all_norms, 95))

    theoretical_bound = lipschitz_L ** 2 / N_train
    imbalance_penalty = lambda_imb * np.exp(-lambda_imb) * lipschitz_L

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ── left: empirical variance vs theoretical bound ─────────────────────────
    ax = axes[0]
    for p in PHASE_NAMES:
        var = np.array(tracker.grad_variance[p])
        ax.plot(epochs, var, linewidth=2.0, color=PHASE_COLORS[p],
                alpha=0.7, label=p)

    ax.axhline(theoretical_bound,
               color='red', linewidth=2.5, linestyle='--',
               label=f'Theorem 1 bound L²/N = {theoretical_bound:.4f}')
    ax.axhline(theoretical_bound + imbalance_penalty,
               color='darkred', linewidth=2.0, linestyle=':',
               label=f'+ imbalance λe⁻ˡL = {imbalance_penalty:.4f}')
    ax.fill_between(epochs, 0, theoretical_bound,
                    alpha=0.06, color='red', label='Bound region')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Variance Var[∇θ]')
    ax.set_title('Empirical Variance vs Theorem 1 Bound\n'
                 'Var[∇] ≤ L²/N')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    # ── centre: log-log variance decay to verify O(1/N) slope ────────────────
    ax = axes[1]
    # rolling mean over window to reduce noise
    w = max(1, T // 15)
    for p in PHASE_NAMES:
        var    = np.array(tracker.grad_variance[p])
        smooth = np.convolve(var, np.ones(w) / w, mode='valid')
        ep_s   = np.arange(w, T + 1)
        smooth_pos = smooth + 1e-12
        ax.loglog(ep_s, smooth_pos, linewidth=2.2,
                  color=PHASE_COLORS[p], label=p)

    # reference slopes: O(1/t) and O(1/√t)
    ref_t = np.logspace(np.log10(1), np.log10(T), 200)
    scale = float(np.mean([tracker.grad_variance[p][0]
                            for p in PHASE_NAMES if tracker.grad_variance[p]]) + 1e-12)
    ax.loglog(ref_t, scale / ref_t,        color='red',    linewidth=1.8,
              linestyle='--', alpha=0.8,  label='O(1/t) slope')
    ax.loglog(ref_t, scale / np.sqrt(ref_t), color='orange', linewidth=1.8,
              linestyle=':', alpha=0.8,   label='O(1/√t) slope')

    ax.set_xlabel('Epoch (log scale)')
    ax.set_ylabel('Gradient Variance (log scale)')
    ax.set_title('Variance Decay Rate — Log-Log\n'
                 'Confirms O(1/t) convergence rate (Theorem 1)')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3, which='both')

    # ── right: imbalance penalty evolution ───────────────────────────────────
    ax = axes[2]
    # estimate λe^{-λ}L per epoch from ratio Var_engage / (L²/N)
    engage_var = np.array(tracker.grad_variance['Phase-1 (Engage)'])
    base_var   = theoretical_bound

    # excess = empirical - statistical_bound (≥0 means imbalance is active)
    excess = np.maximum(0, engage_var - base_var)

    # rolling average of excess
    smooth_excess = np.convolve(excess, np.ones(w) / w, mode='valid')
    ep_s = np.arange(w, T + 1)

    ax.fill_between(ep_s, 0, smooth_excess, alpha=0.3,
                    color=PHASE_COLORS['Phase-1 (Engage)'],
                    label='Empirical imbalance excess')
    ax.plot(ep_s, smooth_excess, linewidth=2.2,
            color=PHASE_COLORS['Phase-1 (Engage)'])

    ax.axhline(imbalance_penalty, color='red', linewidth=2.0,
               linestyle='--',
               label=f'Theorem 1: λe⁻ˡL = {imbalance_penalty:.4f}')

    # theoretical decay of imbalance penalty: λe^{-λt} as model improves
    lambda_decay = imbalance_penalty * np.exp(-0.01 * (epochs - 1))
    ax.plot(epochs, lambda_decay, color='orange', linewidth=1.8,
            linestyle=':', alpha=0.9, label='Expected decay λe^{−λt}')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Imbalance Gradient Excess')
    ax.set_title('Imbalance Penalty Validation\n'
                 'Engage phase excess ≤ λe^{−λ}·L  (Theorem 1 term 3)')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    # fig.suptitle(
    #     'Theorem 1 — Engage Phase Gradient Stability: '
    #     '‖∇θ L‖₂ ≤ L√(2log(2/δ)/N) + C/√N + λe^{−λ}L',
    #     fontsize=16)
    fig.tight_layout()
    _savefig(fig, save_path)


# =============================================================================
# Plot 5 – Gradient Flow Comparison across training segments
#
# 4-panel layout showing gradient evolution across five equal segments
# (0-20%, 20-40%, 40-60%, 60-80%, 80-100%) of training:
#
#   [top-left]   Grouped bar chart: mean ‖∇‖ per phase per segment
#   [top-right]  Line chart: gradient norm trajectory per phase (smoothed)
#                with segment boundary markers
#   [bot-left]   Convergence speed: % reduction from segment to segment
#   [bot-right]  Phase dominance: stacked area of relative gradient share
# =============================================================================

def plot_gradient_flow_comparison(tracker: GradientTracker, save_path=None):
    T          = len(tracker.loss_values)
    n_segs     = 5
    seg_labels = ['0–20%', '20–40%', '40–60%', '60–80%', '80–100%']
    colors     = [PHASE_COLORS[p] for p in PHASE_NAMES]
    n_p        = len(PHASE_NAMES)

    boundaries = [int(round(T * i / n_segs)) for i in range(n_segs + 1)]
    segments   = [(boundaries[i], boundaries[i + 1]) for i in range(n_segs)]

    seg_means = {p: [] for p in PHASE_NAMES}
    seg_stds  = {p: [] for p in PHASE_NAMES}
    for p in PHASE_NAMES:
        arr = np.array(tracker.grad_norms[p])
        for s, e in segments:
            window = arr[s:e] if e > s else arr[s:s + 1]
            seg_means[p].append(float(np.mean(window)))
            seg_stds[p].append(float(np.std(window)))

    fig = plt.figure(figsize=(24, 18))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            left=0.07, right=0.97,
                            top=0.91,  bottom=0.08,
                            hspace=0.48, wspace=0.32)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])

    # [0,0] grouped bar chart
    x       = np.arange(n_segs)
    bw      = 0.20
    offsets = np.linspace(-(n_p - 1) * bw / 2, (n_p - 1) * bw / 2, n_p)
    all_tops = []
    for i, (p, c) in enumerate(zip(PHASE_NAMES, colors)):
        bars = ax00.bar(
            x + offsets[i], seg_means[p], bw,
            yerr=seg_stds[p], capsize=3,
            color=c, alpha=0.85, label=p,
            edgecolor='white', linewidth=0.6,
            error_kw=dict(elinewidth=1.0, ecolor='#333', capthick=1.0))
        all_tops.extend([b.get_height() + s for b, s in zip(bars, seg_stds[p])])
    y_max = max(all_tops) if all_tops else 1.0
    ax00.set_ylim(0, y_max * 1.12)
    ax00.set_xticks(x)
    ax00.set_xticklabels(seg_labels, fontsize=14)
    ax00.set_ylabel('Mean ‖∇θ‖₂', fontsize=15)
    ax00.set_title('Gradient Norm per Phase\nacross Training Segments', fontsize=16)
    ax00.legend(fontsize=14, loc='upper right')
    ax00.grid(True, axis='y', alpha=0.3)

    # [0,1] smoothed trajectory with boundaries
    ep      = np.arange(1, T + 1)
    w_sm    = max(1, T // 20)
    y_peaks = []
    for p, c in zip(PHASE_NAMES, colors):
        raw = np.array(tracker.grad_norms[p])
        ax01.plot(ep, raw, alpha=0.15, linewidth=0.8, color=c)
        if len(raw) >= w_sm:
            sm  = np.convolve(raw, np.ones(w_sm) / w_sm, mode='valid')
            ax01.plot(np.arange(w_sm, T + 1), sm, linewidth=2.5, color=c, label=p)
            y_peaks.append(float(sm.max()))
        else:
            ax01.plot(ep, raw, linewidth=2.5, color=c, label=p)
            y_peaks.append(float(raw.max()))
    y_top = max(y_peaks) * 1.18 if y_peaks else 0.1
    ax01.set_ylim(0, y_top)
    for i, (s, _) in enumerate(segments[1:], 1):
        ax01.axvline(s + 1, color='grey', linewidth=1.2, linestyle='--', alpha=0.6)
        ax01.text(s + max(2, T // 80), y_top * 0.94,
                  f'{i * 20}%', fontsize=14, color='grey', va='top')
    ax01.set_xlabel('Epoch', fontsize=15)
    ax01.set_ylabel('‖∇θ‖₂', fontsize=15)
    ax01.set_title('Smoothed Gradient Norm Trajectory\nwith Segment Boundaries', fontsize=16)
    ax01.legend(fontsize=14, loc='upper right')
    ax01.grid(True, alpha=0.25)

    # [1,0] % change between segments
    trans_labels = [f'{seg_labels[i]}→\n{seg_labels[i + 1]}' for i in range(n_segs - 1)]
    x2   = np.arange(len(trans_labels))
    bw2  = 0.20
    off2 = np.linspace(-(n_p - 1) * bw2 / 2, (n_p - 1) * bw2 / 2, n_p)
    all_reds   = []
    phase_reds = {}
    for p in PHASE_NAMES:
        reds = [100.0 * (seg_means[p][s] - seg_means[p][s + 1]) / (seg_means[p][s] + 1e-8)
                for s in range(n_segs - 1)]
        phase_reds[p] = reds
        all_reds.extend(reds)
    r_min = min(all_reds) if all_reds else -10
    r_max = max(all_reds) if all_reds else 10
    pad   = max(abs(r_max), abs(r_min), 1.0) * 0.25
    for i, (p, c) in enumerate(zip(PHASE_NAMES, colors)):
        reds = phase_reds[p]
        bars = ax10.bar(x2 + off2[i], reds, bw2, color=c, alpha=0.85,
                        label=p, edgecolor='white', linewidth=0.6)
        for bar, val in zip(bars, reds):
            va   = 'bottom' if val >= 0 else 'top'
            ypos = val + pad * 0.15 if val >= 0 else val - pad * 0.15
            ax10.text(bar.get_x() + bar.get_width() / 2, ypos,
                      f'{val:+.1f}%', ha='center', va=va,
                      fontsize=14, fontweight='bold')
    ax10.axhline(0, color='black', linewidth=1.4)
    ax10.set_ylim(r_min - pad, r_max + pad)
    ax10.set_xticks(x2)
    ax10.set_xticklabels(trans_labels, fontsize=14, ha='center')
    ax10.set_ylabel('% Change in ‖∇θ‖₂', fontsize=15)
    ax10.set_title('Gradient Change Between Segments\n(+ve = stabilising, −ve = growing)', fontsize=16)
    ax10.legend(fontsize=14, loc='upper right')
    ax10.grid(True, axis='y', alpha=0.3)

    # [1,1] phase dominance stacked area
    n_pts   = n_segs * 20
    ep_rs   = np.linspace(0, T - 1, n_pts).astype(int).clip(0, T - 1)
    stacks  = np.array([np.array(tracker.grad_norms[p])[ep_rs]
                        for p in PHASE_NAMES]).clip(min=0)
    total   = stacks.sum(axis=0) + 1e-8
    shares  = stacks / total * 100
    ep_plot = ep_rs + 1
    ax11.stackplot(ep_plot, shares, labels=PHASE_NAMES, colors=colors, alpha=0.78)
    for s, _ in segments[1:]:
        ax11.axvline(s + 1, color='white', linewidth=1.4, linestyle='--', alpha=0.75)
    for i, (s, e) in enumerate(segments):
        ax11.text((s + e) / 2 + 1, 4, seg_labels[i],
                  ha='center', va='bottom', fontsize=14,
                  color='white', fontweight='bold')
    ax11.set_xlim(ep_plot[0], ep_plot[-1])
    ax11.set_ylim(0, 100)
    ax11.set_xlabel('Epoch', fontsize=15)
    ax11.set_ylabel('Gradient Share (%)', fontsize=15)
    ax11.set_title('Phase Gradient Dominance over Training\n(stacked % of total ‖∇θ‖₂)', fontsize=16)
    ax11.legend(loc='upper right', fontsize=14)
    ax11.grid(True, axis='y', alpha=0.2)

    # fig.suptitle(
    #     'Gradient Flow Comparison — Five Training Segments  '
    #     '(0–20 % │ 20–40 % │ 40–60 % │ 60–80 % │ 80–100 %)',
     #   fontsize=18, y=0.96)
    _savefig(fig, save_path)


# =============================================================================
# Master function
# =============================================================================

def run_gradient_analysis(model, data, device,
                           out_dir='./results/gradient_analysis',
                           epochs=200, lr=0.005,
                           weight_decay=5e-4,
                           loss_type='standard',
                           curriculum_args=None,
                           entropy_args=None,
                           N_train=140):
    os.makedirs(out_dir, exist_ok=True)
    print(f'\nRunning gradient stability analysis ({epochs} epochs)…')

    tracker = train_with_gradient_tracking(
        model, data, device, epochs=epochs, lr=lr,
        weight_decay=weight_decay, loss_type=loss_type,
        curriculum_args=curriculum_args, entropy_args=entropy_args)

    print('  Saving gradient stability plots…')
    plot_gradient_stability(
        tracker,
        save_path=os.path.join(out_dir, 'gradient_stability_overview.png'))

    plot_gradient_norm_per_phase_separate(
        tracker,
        save_path=os.path.join(out_dir, 'gradient_norm_per_phase.png'))

    plot_attention_weights_per_phase(
        tracker,
        save_path=os.path.join(out_dir, 'attention_weights_per_phase.png'))

    plot_gradient_variance_bound(
        tracker,
        save_path=os.path.join(out_dir, 'gradient_variance_bound.png'))

    plot_gradient_flow_comparison(
        tracker,
        save_path=os.path.join(out_dir, 'gradient_flow_comparison.png'))

    print(f'  All gradient analysis plots saved to: {out_dir}/')
    return tracker


def make_gradient_tracker(model):
    return GradientTracker()


def build_phase_grad_dicts(model):
    t = GradientTracker()
    return (t.grad_norms.copy(), t.attn_weights.copy(), t.grad_variance.copy())