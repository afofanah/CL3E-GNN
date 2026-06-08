import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.transforms import NormalizeFeatures
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import Evaluator
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
from models.model import CL3E_ModelV1
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import seaborn as sns
from sklearn.manifold import TSNE
import argparse
import random
import os
import gc

FONT_MIN = 15
FONT_AXIS = 16

def save_fig(path_no_ext):
    plt.savefig(f'{path_no_ext}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{path_no_ext}.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Graph Neural Network Training')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--feat_dim', type=int, default=512)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--out_dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5)
    parser.add_argument('--lr_scheduler_patience', type=int, default=10)
    return parser.parse_args([])

args = parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


dataset = PygNodePropPredDataset(root='data/ogbn-arxiv', name='ogbn-arxiv')
data = dataset[0]

print(f"Nodes: {data.num_nodes}, Features: {dataset.num_features}, Classes: {dataset.num_classes}")

loader = DataLoader([data], batch_size=1, shuffle=False)
model = CL3E_ModelV1(num_features=dataset.num_features, hidden_dim=64, num_classes=dataset.num_classes)

optimizer = Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

best_val_accuracy = 0.0
patience = 10
counter = 0

num_nodes = data.num_nodes
train_val_indices, test_indices = train_test_split(range(num_nodes), test_size=0.2, random_state=42)
train_indices, val_indices = train_test_split(train_val_indices, test_size=0.1, random_state=42)

data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.train_mask[train_indices] = True
data.val_mask[val_indices] = True
data.test_mask[test_indices] = True

train_losses, val_losses, train_accuracies, val_accuracies, auc_scores = [], [], [], [], []

gradient_norm_history = {'Phase_1': [], 'Phase_2': [], 'Phase_3': []}
attention_weight_history = {'Phase_1': [], 'Phase_2': [], 'Phase_3': []}
grad_variance_history = {'Phase_1': [], 'Phase_2': [], 'Phase_3': []}
variance_window = 10

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    output, _ = model(data.x, data.edge_index)
    train_output = output[data.train_mask]
    train_labels = data.y[data.train_mask]
    if train_labels.dim() > 1:
        train_labels = train_labels.squeeze()
    train_labels = train_labels.long()
    loss = criterion(train_output, train_labels)

    loss.backward()

    phase_1_grads, phase_2_grads, phase_3_grads = [], [], []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if 'phase_1' in name or 'Phase_1' in name:
                phase_1_grads.append(param.grad.norm().item())
            elif 'phase_2' in name or 'Phase_2' in name:
                phase_2_grads.append(param.grad.norm().item())
            elif 'phase_3' in name or 'Phase_3' in name:
                phase_3_grads.append(param.grad.norm().item())

    gradient_norm_history['Phase_1'].append(np.mean(phase_1_grads) if phase_1_grads else 0)
    gradient_norm_history['Phase_2'].append(np.mean(phase_2_grads) if phase_2_grads else 0)
    gradient_norm_history['Phase_3'].append(np.mean(phase_3_grads) if phase_3_grads else 0)

    with torch.no_grad():
        emb = model.embedding_layer(data.x)
        for i in range(model.num_gcn_layers):
            emb = F.relu(model.gcn_layers[i](emb, data.edge_index))
            emb = model.dropout(emb)
        attention_weight_history['Phase_1'].append(torch.sigmoid(model.stage_1_attention(emb)).mean().item())
        attention_weight_history['Phase_2'].append(torch.sigmoid(model.stage_2_attention(emb)).mean().item())
        attention_weight_history['Phase_3'].append(torch.sigmoid(model.stage_3_attention(emb)).mean().item())

    if epoch >= variance_window:
        for ph in ['Phase_1', 'Phase_2', 'Phase_3']:
            grad_variance_history[ph].append(np.var(gradient_norm_history[ph][-variance_window:]))

    optimizer.step()

    train_losses.append(loss.item())
    train_accuracies.append(accuracy_score(train_labels.cpu().numpy(), train_output.argmax(dim=1).detach().cpu().numpy()) * 100)

    model.eval()
    with torch.no_grad():
        val_output = output[data.val_mask]
        val_labels = data.y[data.val_mask]
        if val_labels.dim() > 1:
            val_labels = val_labels.squeeze()
        val_labels = val_labels.long()

        val_losses.append(criterion(val_output, val_labels).item())
        val_accuracies.append(accuracy_score(val_labels.cpu().numpy(), val_output.argmax(dim=1).cpu().numpy()) * 100)

        val_probs = F.softmax(val_output, dim=1).cpu().numpy()
        val_true = val_labels.cpu().numpy()
        try:
            auc_scores.append(roc_auc_score(np.eye(dataset.num_classes)[val_true], val_probs, multi_class='ovr'))
        except Exception as e:
            print(f"AUC calculation error: {e}")
            auc_scores.append(0)

    print(f'Epoch: {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, '
          f'Train Acc: {train_accuracies[-1]:.2f}%, Val Acc: {val_accuracies[-1]:.2f}%, '
          f'AUC: {auc_scores[-1]:.4f}')

    scheduler.step(val_accuracies[-1])

    if val_accuracies[-1] > best_val_accuracy:
        best_val_accuracy = val_accuracies[-1]
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping.")
            break

# ---- Training curves ----
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
ax.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
ax.set_xlabel('Epoch', fontsize=FONT_AXIS)
ax.set_ylabel('Loss', fontsize=FONT_AXIS)
ax.set_title('Training and Validation Loss', fontsize=FONT_MIN)
ax.legend(fontsize=FONT_AXIS)
ax.tick_params(axis='both', labelsize=FONT_MIN)
ax.grid(True)
plt.tight_layout()
save_fig('plots/loss_curves')

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', color='blue')
ax.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='green')
ax.set_xlabel('Epoch', fontsize=FONT_AXIS)
ax.set_ylabel('Accuracy (%)', fontsize=FONT_AXIS)
ax.set_title('Train and Validation Accuracies', fontsize=FONT_MIN)
ax.legend(fontsize=FONT_AXIS)
ax.tick_params(axis='both', labelsize=FONT_MIN)
ax.grid(True)
plt.tight_layout()
save_fig('plots/accuracy_curves')

if auc_scores:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(auc_scores) + 1), auc_scores, label='AUC Score', color='orange')
    ax.set_xlabel('Epoch', fontsize=FONT_AXIS)
    ax.set_ylabel('Score', fontsize=FONT_AXIS)
    ax.set_title('AUC Score over Epochs', fontsize=FONT_MIN)
    ax.legend(fontsize=FONT_AXIS)
    ax.tick_params(axis='both', labelsize=FONT_MIN)
    ax.grid(True)
    plt.tight_layout()
    save_fig('plots/auc_evolution')

# ---- Test evaluation ----
model.eval()
with torch.no_grad():
    test_output = output[data.test_mask]
    test_labels = data.y[data.test_mask]
    if test_labels.dim() > 1:
        test_labels = test_labels.squeeze()
    test_labels = test_labels.long()
    test_pred = test_output.argmax(dim=1)
    test_accuracy = accuracy_score(test_labels.cpu().numpy(), test_pred.detach().cpu().numpy())
    test_probs = F.softmax(test_output, dim=1).cpu().numpy()
    test_true = test_labels.cpu().numpy()
    f1_score_macro = f1_score(test_true, test_pred.cpu().numpy(), average='macro')
    f1_score_micro = f1_score(test_true, test_pred.cpu().numpy(), average='micro')
    f1_score_weighted = f1_score(test_true, test_pred.cpu().numpy(), average='weighted')

# ============= ROC CURVES =============
print("\nCalculating ROC curves...")
model.eval()
with torch.no_grad():
    output, _ = model(data.x, data.edge_index)
    test_output = output[data.test_mask]
    test_labels = data.y[data.test_mask]
    if test_labels.dim() > 1:
        test_labels = test_labels.squeeze()
    test_labels = test_labels.long()
    test_probs = F.softmax(test_output, dim=1).cpu().numpy()
    test_true = test_labels.cpu().numpy()

fpr, tpr, roc_auc = dict(), dict(), dict()
test_true_one_hot = np.zeros((test_true.size, dataset.num_classes))
for i in range(test_true.size):
    test_true_one_hot[i, test_true[i]] = 1

for i in range(dataset.num_classes):
    try:
        fpr[i], tpr[i], _ = roc_curve(test_true_one_hot[:, i], test_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        if np.isnan(roc_auc[i]) or len(fpr[i]) < 2:
            fpr[i], tpr[i] = np.array([0, 1]), np.array([0, 1])
            roc_auc[i] = 0.5
    except Exception as e:
        print(f"Error calculating ROC for class {i}: {e}")
        fpr[i], tpr[i] = np.array([0, 1]), np.array([0, 1])
        roc_auc[i] = 0.5

avg_auc = np.mean(list(roc_auc.values()))
print(f"Average AUC across all classes: {avg_auc:.4f}")

save_dir = 'roc_plots'
os.makedirs(save_dir, exist_ok=True)

# Multi-subplot ROC (all classes)
num_classes = dataset.num_classes
classes_per_plot = num_classes // 4 + (1 if num_classes % 4 > 0 else 0)

fig_all, axes_all = plt.subplots(2, 2, figsize=(20, 16))
axes_all = axes_all.flatten()
for subplot_idx in range(4):
    ax = axes_all[subplot_idx]
    start_class = subplot_idx * classes_per_plot
    end_class = min((subplot_idx + 1) * classes_per_plot, num_classes)

    for i in range(start_class, end_class):
        if i < num_classes and roc_auc[i] > 0:
            ax.plot(fpr[i], tpr[i], lw=1.5, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    ax.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=FONT_AXIS)
    ax.set_ylabel('True Positive Rate', fontsize=FONT_AXIS)
    ax.set_title(f'ROC Curves: Classes {start_class}–{end_class-1}', fontsize=FONT_MIN)
    ax.tick_params(axis='both', labelsize=FONT_MIN)
    ax.grid(True, alpha=0.3)

    if end_class - start_class > 10:
        group_aucs = {i: roc_auc[i] for i in range(start_class, end_class) if i < num_classes}
        top_class_indices = [cls for cls, _ in sorted(group_aucs.items(), key=lambda x: x[1], reverse=True)[:10]]
        handles, labels = ax.get_legend_handles_labels()
        fh, fl = [], []
        for i, label in enumerate(labels):
            if 'Class' in label:
                try:
                    if int(label.split(' ')[1]) in top_class_indices:
                        fh.append(handles[i]); fl.append(label)
                except (ValueError, IndexError):
                    pass
        ax.legend(fh, fl, loc='lower right', fontsize=FONT_AXIS)
    else:
        ax.legend(loc='lower right', fontsize=FONT_AXIS)

plt.tight_layout()
fig_all.suptitle('ROC Curves for All Classes', fontsize=FONT_MIN + 2, y=1.02)
fig_all.savefig(os.path.join(save_dir, 'roc_curves_all_classes.png'), dpi=300, bbox_inches='tight')
fig_all.savefig(os.path.join(save_dir, 'roc_curves_all_classes.pdf'), dpi=300, bbox_inches='tight')
plt.close(fig_all)

# Top 10 classes ROC
top_class_indices_10 = [cls for cls, _ in sorted(roc_auc.items(), key=lambda x: x[1], reverse=True)[:10]]
fig, ax = plt.subplots(figsize=(12, 10))
for i in top_class_indices_10:
    ax.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=FONT_AXIS)
ax.set_ylabel('True Positive Rate', fontsize=FONT_AXIS)
ax.set_title('ROC Curves for Top 10 Classes by AUC Score', fontsize=FONT_MIN)
ax.legend(loc='lower right', fontsize=FONT_AXIS)
ax.tick_params(axis='both', labelsize=FONT_MIN)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig(os.path.join(save_dir, 'top_roc_curves'))

# Top 20 vs Bottom 20
sorted_aucs = sorted(roc_auc.items(), key=lambda x: x[1], reverse=True)
top_20_classes = [cls for cls, _ in sorted_aucs[:20]]
bottom_20_classes = [cls for cls, _ in sorted_aucs[-20:]]

fig_top_bottom, (ax_top, ax_bottom) = plt.subplots(1, 2, figsize=(20, 10))
for i in top_20_classes:
    ax_top.plot(fpr[i], tpr[i], lw=1.5, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
ax_top.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)
ax_top.set_xlim([0.0, 1.0]); ax_top.set_ylim([0.0, 1.05])
ax_top.set_xlabel('False Positive Rate', fontsize=FONT_AXIS)
ax_top.set_ylabel('True Positive Rate', fontsize=FONT_AXIS)
ax_top.set_title('ROC Curves: Top 20 Classes by AUC', fontsize=FONT_MIN)
ax_top.tick_params(axis='both', labelsize=FONT_MIN)
ax_top.grid(True, alpha=0.3)
handles, labels = ax_top.get_legend_handles_labels()
ax_top.legend(handles, labels, loc='lower right', fontsize=FONT_AXIS, ncol=2)

for i in bottom_20_classes:
    ax_bottom.plot(fpr[i], tpr[i], lw=1.5, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
ax_bottom.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)
ax_bottom.set_xlim([0.0, 1.0]); ax_bottom.set_ylim([0.0, 1.05])
ax_bottom.set_xlabel('False Positive Rate', fontsize=FONT_AXIS)
ax_bottom.set_ylabel('True Positive Rate', fontsize=FONT_AXIS)
ax_bottom.set_title('ROC Curves: Bottom 20 Classes by AUC', fontsize=FONT_MIN)
ax_bottom.tick_params(axis='both', labelsize=FONT_MIN)
ax_bottom.grid(True, alpha=0.3)
handles, labels = ax_bottom.get_legend_handles_labels()
ax_bottom.legend(handles, labels, loc='lower right', fontsize=FONT_AXIS, ncol=2)

plt.tight_layout()
fig_top_bottom.suptitle('Comparison of Top 20 vs Bottom 20 Classes by AUC', fontsize=FONT_MIN + 2, y=1.02)
fig_top_bottom.savefig(os.path.join(save_dir, 'roc_curves_top_bottom_20.png'), dpi=300, bbox_inches='tight')
fig_top_bottom.savefig(os.path.join(save_dir, 'roc_curves_top_bottom_20.pdf'), dpi=300, bbox_inches='tight')
plt.close(fig_top_bottom)

# AUC bar chart
classes = np.arange(dataset.num_classes)
auc_values = [roc_auc[i] for i in classes]
sorted_indices = np.argsort(auc_values)[::-1]
sorted_classes = classes[sorted_indices]
sorted_auc_values = [auc_values[i] for i in sorted_indices]

bar_colors = ['lightblue'] * len(sorted_classes)
for i in range(20):
    bar_colors[i] = 'green'
for i in range(1, 21):
    bar_colors[-i] = 'red'

fig, ax = plt.subplots(figsize=(14, 8))
ax.bar(np.arange(len(sorted_classes)), sorted_auc_values, align='center', alpha=0.7, color=bar_colors)
ax.set_xlabel('Class Index (Sorted by AUC)', fontsize=FONT_AXIS)
ax.set_ylabel('AUC Score', fontsize=FONT_AXIS)
ax.set_title('AUC Scores for All Classes', fontsize=FONT_MIN)
ax.axhline(y=avg_auc, color='black', linestyle='--', label=f'Avg AUC = {avg_auc:.2f}')
ax.tick_params(axis='both', labelsize=FONT_MIN)
ax.grid(True, axis='y', alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', label='Top 20 Classes'),
    Patch(facecolor='red', label='Bottom 20 Classes'),
    Patch(facecolor='lightblue', label='Other Classes')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=FONT_AXIS)

step = max(1, len(sorted_classes) // 20)
ax.set_xticks(np.arange(0, len(sorted_classes), step))
ax.set_xticklabels([sorted_classes[i] for i in range(0, len(sorted_classes), step)], rotation=45, fontsize=FONT_MIN)
plt.tight_layout()
save_fig(os.path.join(save_dir, 'auc_scores_barchart'))

print(f"ROC curve plots saved to {save_dir}/")
print("\nTop 20 Classes by AUC:")
for i, (cls, score) in enumerate(sorted_aucs[:20], 1):
    print(f"{i}. Class {cls}: AUC = {score:.4f}")
print("\nBottom 20 Classes by AUC:")
for i, (cls, score) in enumerate(sorted_aucs[-20:], 1):
    print(f"{i}. Class {cls}: AUC = {score:.4f}")

# Class metrics CSV
class_metrics = []
for i in range(dataset.num_classes):
    binary_true = (test_true == i).astype(int)
    binary_pred = (test_probs[:, i] >= 0.5).astype(int)
    count = np.sum(binary_true)
    if count > 0:
        try:
            f1 = f1_score(binary_true, binary_pred)
            class_metrics.append({'class': i, 'count': count, 'auc': roc_auc[i], 'f1': f1})
        except Exception as e:
            print(f"Error calculating metrics for class {i}: {e}")

import pandas as pd
pd.DataFrame(class_metrics).to_csv('class_metrics.csv', index=False)

overall_auc = np.mean(list(roc_auc.values()))
print(f'\nFinal AUC Score: {auc_scores[-1] if auc_scores else 0:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Overall Test AUC: {overall_auc:.4f}')
print(f'Test F1-score (Macro): {f1_score_macro:.4f}')
print(f'Test F1-score (Micro): {f1_score_micro:.4f}')
print(f'Test F1-score (Weighted): {f1_score_weighted:.4f}')

# ============= ATTENTION WEIGHTS =============
model.eval()
with torch.no_grad():
    output, combined_representation = model(data.x, data.edge_index)
    embeddings = model.embedding_layer(data.x)
    for i in range(model.num_gcn_layers):
        embeddings = F.relu(model.gcn_layers[i](embeddings, data.edge_index))
        embeddings = model.dropout(embeddings)

    phase_1_attn = torch.sigmoid(model.stage_1_attention(embeddings)).detach().cpu().numpy().flatten()
    phase_2_attn = torch.sigmoid(model.stage_2_attention(embeddings)).detach().cpu().numpy().flatten()
    phase_3_attn = torch.sigmoid(model.stage_3_attention(embeddings)).detach().cpu().numpy().flatten()

sample_size = min(100, data.num_nodes)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    np.array([phase_1_attn[:sample_size], phase_2_attn[:sample_size], phase_3_attn[:sample_size]]),
    cmap='Spectral', xticklabels=True, yticklabels=['Phase 1', 'Phase 2', 'Phase 3'], ax=ax
)
ax.set_title('Attention Weights Heatmap (First 100 Nodes)', fontsize=FONT_MIN)
ax.set_xlabel('Node Index', fontsize=FONT_AXIS)
ax.set_ylabel('Attention Phase', fontsize=FONT_AXIS)
ax.tick_params(axis='both', labelsize=FONT_MIN)
plt.tight_layout()
save_fig('plots/attention_heatmap')

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(['Phase 1', 'Phase 2', 'Phase 3'],
       [phase_1_attn.sum(), phase_2_attn.sum(), phase_3_attn.sum()],
       color=['blue', 'orange', 'green'])
ax.set_title('Aggregated Attention Weights per Phase', fontsize=FONT_MIN)
ax.set_xlabel('Attention Phase', fontsize=FONT_AXIS)
ax.set_ylabel('Total Attention Weight', fontsize=FONT_AXIS)
ax.tick_params(axis='both', labelsize=FONT_MIN)
plt.tight_layout()
save_fig('plots/attention_bar')

# ============= t-SNE =============
print("\nExtracting features for t-SNE...")
model.eval()
with torch.no_grad():
    output, _ = model(data.x, data.edge_index)
    sample_size = min(1000, data.num_nodes)
    test_idx = torch.where(data.test_mask)[0].numpy()
    if len(test_idx) > sample_size:
        sample_indices = np.random.choice(test_idx, sample_size, replace=False)
    else:
        remaining = sample_size - len(test_idx)
        other_idx = torch.where(~data.test_mask)[0].numpy()
        extra = np.random.choice(other_idx, min(remaining, len(other_idx)), replace=False) if len(other_idx) > 0 else np.array([])
        sample_indices = np.concatenate([test_idx, extra]).astype(int)

    sample_embeddings = output[sample_indices].numpy()
    sample_labels = data.y[sample_indices].numpy()

print("Computing t-SNE embedding...")
try:
    tsne = TSNE(n_components=2, random_state=args.seed, perplexity=min(30, len(sample_indices) - 1))
    embeddings_2d = tsne.fit_transform(sample_embeddings)
except Exception as e:
    print(f"t-SNE error: {e}")
    embeddings_2d = None

if embeddings_2d is not None:
    from matplotlib.colors import ListedColormap
    import matplotlib.cm as cm
    base_colors = plt.colormaps['tab20'](np.linspace(0, 1, 20))
    base_colors2 = plt.colormaps['tab20b'](np.linspace(0, 1, 20))
    all_colors = np.vstack([base_colors, base_colors2])

    classes_per_plot = num_classes // 4
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    for subplot_idx in range(4):
        ax = axes[subplot_idx]
        start_class = subplot_idx * classes_per_plot
        end_class = min((subplot_idx + 1) * classes_per_plot, num_classes)
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='lightgray', alpha=0.3, s=20)
        for class_idx in range(start_class, end_class):
            idx = np.where(sample_labels == class_idx)[0]
            if len(idx) > 0:
                ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1],
                           label=f'Class {class_idx}', color=all_colors[class_idx % all_colors.shape[0]],
                           alpha=0.8, s=50)
        ax.set_title(f't-SNE Visualization (Classes {start_class}–{end_class-1}, fontsize=15)', fontsize=FONT_MIN)
        ax.set_xlabel('t-SNE Component 1', fontsize=FONT_AXIS)
        ax.set_ylabel('t-SNE Component 2', fontsize=FONT_AXIS)
        ax.tick_params(axis='both', labelsize=FONT_MIN)
        ax.grid(False)
        handles, labels = ax.get_legend_handles_labels()
        shown = handles[:10] if len(handles) > 10 else handles
        shown_l = labels[:10] if len(labels) > 10 else labels
        ax.legend(shown, shown_l, loc='upper right', fontsize=FONT_AXIS)

    plt.tight_layout()
    fig.savefig('plots/tsne_visualization_multi.png', dpi=300, bbox_inches='tight')
    fig.savefig('plots/tsne_visualization_multi.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 12))
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=sample_labels, cmap=ListedColormap(all_colors), alpha=0.7, s=50)
    cbar = plt.colorbar(scatter, ax=ax, label='Class Label')
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('Class Label', fontsize=FONT_AXIS)
    tick_step = num_classes // 10 + 1
    cbar.set_ticks((np.arange(0, num_classes, tick_step) + 0.5) * (num_classes - 1) / num_classes)
    cbar.set_ticklabels(np.arange(0, num_classes, tick_step))
    ax.set_title('t-SNE Visualization of All Classes', fontsize=FONT_MIN)
    ax.set_xlabel('t-SNE Component 1', fontsize=FONT_AXIS)
    ax.set_ylabel('t-SNE Component 2', fontsize=FONT_AXIS)
    ax.tick_params(axis='both', labelsize=FONT_MIN)
    ax.grid(False)
    plt.tight_layout()
    save_fig('plots/tsne_visualization_all')

# ============= CONFUSION MATRICES =============
test_pred_np = test_pred.numpy()
test_labels_np = test_labels.numpy()
class_counts = np.bincount(test_labels_np, minlength=dataset.num_classes)

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(np.arange(dataset.num_classes), class_counts)
ax.set_xlabel('Class', fontsize=FONT_AXIS)
ax.set_ylabel('Number of Nodes', fontsize=FONT_AXIS)
ax.set_title('Class Distribution in Test Set', fontsize=FONT_MIN)
ax.set_xticks(np.arange(dataset.num_classes)[::5])
ax.set_xticklabels(np.arange(dataset.num_classes)[::5], fontsize=FONT_MIN)
ax.tick_params(axis='both', labelsize=FONT_MIN)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
save_fig('plots/class_distribution')

top_classes_cm = np.argsort(class_counts)[-40:]
num_groups = 4
classes_per_group = 10

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()
for group_idx in range(num_groups):
    start_idx = group_idx * classes_per_group
    end_idx = min((group_idx + 1) * classes_per_group, len(top_classes_cm))
    group_classes = top_classes_cm[start_idx:end_idx]
    mask = np.isin(test_pred_np, group_classes) & np.isin(test_labels_np, group_classes)
    indices = np.where(mask)[0]

    if len(indices) > 0:
        class_mapping = {cls: i for i, cls in enumerate(group_classes)}
        mapped_pred = np.array([class_mapping.get(cls, -1) for cls in test_pred_np[indices]])
        mapped_true = np.array([class_mapping.get(cls, -1) for cls in test_labels_np[indices]])
        cm_group = confusion_matrix(mapped_true, mapped_pred, labels=range(len(group_classes)))
        ax = axes[group_idx]
        sns.heatmap(cm_group, annot=len(group_classes) <= 10, fmt="d", cmap="Blues",
                    xticklabels=[f"{cls}" for cls in group_classes],
                    yticklabels=[f"{cls}" for cls in group_classes], ax=ax)
        ax.set_xlabel('Predicted Label', fontsize=FONT_AXIS)
        ax.set_ylabel('True Label', fontsize=FONT_AXIS)
        ax.set_title(f'Confusion Matrix: Classes {start_idx*10}–{end_idx*10-1}', fontsize=FONT_MIN)
        ax.tick_params(axis='both', labelsize=FONT_MIN)
    else:
        axes[group_idx].text(0.5, 0.5, 'No data for these classes', ha='center', va='center', fontsize=FONT_MIN)

plt.tight_layout()
fig.savefig('plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
fig.savefig('plots/confusion_matrices.pdf', dpi=300, bbox_inches='tight')
plt.close(fig)

try:
    overall_cm = confusion_matrix(test_labels_np, test_pred_np)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(overall_cm, annot=False, fmt="d", cmap="Reds", ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=FONT_AXIS)
    ax.set_ylabel('True Label', fontsize=FONT_AXIS)
    ax.set_title('Overall Confusion Matrix on Test Set', fontsize=FONT_MIN)
    ax.tick_params(axis='both', labelsize=FONT_MIN)
    plt.tight_layout()
    save_fig('plots/full_confusion_matrix')
except Exception as e:
    print(f"Could not generate full confusion matrix: {e}")


# ============= GRADIENT STABILITY ANALYSIS =============
epochs      = range(1, len(gradient_norm_history['Phase_1']) + 1)
var_epochs  = range(variance_window, len(gradient_norm_history['Phase_1']) + 1)
colors = {'Phase_1': '#ff7300', 'Phase_2': '#0088fe', 'Phase_3': '#00c49f'}

os.makedirs('plots', exist_ok=True)

# 1. Gradient norm evolution
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs, gradient_norm_history['Phase_1'], label='Phase 1 (Initial)',         color=colors['Phase_1'], linewidth=2)
ax.plot(epochs, gradient_norm_history['Phase_2'], label='Phase 2 (Attention)',        color=colors['Phase_2'], linewidth=2)
ax.plot(epochs, gradient_norm_history['Phase_3'], label='Phase 3 (Full Integration)', color=colors['Phase_3'], linewidth=2)
ax.set_xlabel('Epoch', fontsize=FONT_AXIS)
ax.set_ylabel('Gradient Norm', fontsize=FONT_AXIS)
ax.set_title('Evolution of Gradient Norms During Training', fontsize=FONT_MIN)
ax.legend(fontsize=FONT_AXIS)
ax.tick_params(axis='both', labelsize=FONT_MIN)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig('plots/gradient_norm_evolution')

# 2. Gradient variance
if len(grad_variance_history['Phase_1']) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(var_epochs, grad_variance_history['Phase_1'], label='Phase 1 Variance', color=colors['Phase_1'])
    ax.plot(var_epochs, grad_variance_history['Phase_2'], label='Phase 2 Variance', color=colors['Phase_2'])
    ax.plot(var_epochs, grad_variance_history['Phase_3'], label='Phase 3 Variance', color=colors['Phase_3'])
    ax.set_xlabel('Epoch', fontsize=FONT_AXIS)
    ax.set_ylabel('Gradient Variance', fontsize=FONT_AXIS)
    ax.set_title('Gradient Variance Reduction During Training', fontsize=FONT_MIN)
    ax.legend(fontsize=FONT_AXIS)
    ax.tick_params(axis='both', labelsize=FONT_MIN)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig('plots/gradient_variance')

# 3. Attention weight evolution
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs, attention_weight_history['Phase_1'], label='Phase 1 Attention', color=colors['Phase_1'])
ax.plot(epochs, attention_weight_history['Phase_2'], label='Phase 2 Attention', color=colors['Phase_2'])
ax.plot(epochs, attention_weight_history['Phase_3'], label='Phase 3 Attention', color=colors['Phase_3'])
ax.set_xlabel('Epoch', fontsize=FONT_AXIS)
ax.set_ylabel('Attention Weight', fontsize=FONT_AXIS)
ax.set_title('Evolution of Attention Weights During Training', fontsize=FONT_MIN)
ax.legend(fontsize=FONT_AXIS)
ax.tick_params(axis='both', labelsize=FONT_MIN)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig('plots/attention_weights')

# 4. Gradient stability (inverse normalised variance)
if len(grad_variance_history['Phase_1']) > 0:
    s1_stab = [1 - min(1, v * 15) for v in grad_variance_history['Phase_1']]
    s2_stab = [1 - min(1, v * 20) for v in grad_variance_history['Phase_2']]
    s3_stab = [1 - min(1, v * 30) for v in grad_variance_history['Phase_3']]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(var_epochs, s1_stab, alpha=0.3, color=colors['Phase_1'], label='Phase 1 Stability')
    ax.fill_between(var_epochs, s2_stab, alpha=0.3, color=colors['Phase_2'], label='Phase 2 Stability')
    ax.fill_between(var_epochs, s3_stab, alpha=0.3, color=colors['Phase_3'], label='Phase 3 Stability')
    ax.axhline(y=0.75, color='red', linestyle='--', label='Stability Threshold')
    ax.set_xlabel('Epoch', fontsize=FONT_AXIS)
    ax.set_ylabel('Gradient Stability (1 – Normalised Variance)', fontsize=FONT_AXIS)
    ax.set_title('Attention Weights vs Gradient Stability', fontsize=FONT_MIN)
    ax.legend(fontsize=FONT_AXIS)
    ax.tick_params(axis='both', labelsize=FONT_MIN)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig('plots/gradient_stability')

# 5. Combined 2x2 gradient analysis panel
if len(grad_variance_history['Phase_1']) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CL3E Model: Gradient Stability Analysis', fontsize=FONT_MIN + 2)

    axes[0, 0].plot(epochs, gradient_norm_history['Phase_1'], label='Phase 1', color=colors['Phase_1'], linewidth=2)
    axes[0, 0].plot(epochs, gradient_norm_history['Phase_2'], label='Phase 2', color=colors['Phase_2'], linewidth=2)
    axes[0, 0].plot(epochs, gradient_norm_history['Phase_3'], label='Phase 3', color=colors['Phase_3'], linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=FONT_AXIS)
    axes[0, 0].set_ylabel('Gradient Norm', fontsize=FONT_AXIS)
    axes[0, 0].set_title('Gradient Norm Evolution', fontsize=FONT_MIN)
    axes[0, 0].legend(fontsize=FONT_AXIS)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='both', labelsize=FONT_MIN)

    axes[0, 1].plot(var_epochs, grad_variance_history['Phase_1'], label='Phase 1', color=colors['Phase_1'])
    axes[0, 1].plot(var_epochs, grad_variance_history['Phase_2'], label='Phase 2', color=colors['Phase_2'])
    axes[0, 1].plot(var_epochs, grad_variance_history['Phase_3'], label='Phase 3', color=colors['Phase_3'])
    axes[0, 1].set_xlabel('Epoch', fontsize=FONT_AXIS)
    axes[0, 1].set_ylabel('Gradient Variance', fontsize=FONT_AXIS)
    axes[0, 1].set_title('Gradient Variance Reduction', fontsize=FONT_MIN)
    axes[0, 1].legend(fontsize=FONT_AXIS)
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='both', labelsize=FONT_MIN)

    axes[1, 0].plot(epochs, attention_weight_history['Phase_1'], label='Phase 1', color=colors['Phase_1'])
    axes[1, 0].plot(epochs, attention_weight_history['Phase_2'], label='Phase 2', color=colors['Phase_2'])
    axes[1, 0].plot(epochs, attention_weight_history['Phase_3'], label='Phase 3', color=colors['Phase_3'])
    axes[1, 0].set_xlabel('Epoch', fontsize=FONT_AXIS)
    axes[1, 0].set_ylabel('Attention Weight', fontsize=FONT_AXIS)
    axes[1, 0].set_title('Attention Weight Evolution', fontsize=FONT_MIN)
    axes[1, 0].legend(fontsize=FONT_AXIS)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='both', labelsize=FONT_MIN)

    sample_idx = list(range(0, len(list(epochs)), 10))
    sample_ep  = [list(epochs)[i] for i in sample_idx]
    s1 = [attention_weight_history['Phase_1'][i] for i in sample_idx]
    s2 = [attention_weight_history['Phase_2'][i] for i in sample_idx]
    s3 = [attention_weight_history['Phase_3'][i] for i in sample_idx]
    bar_idx = np.arange(len(sample_ep))

    axes[1, 1].bar(bar_idx, s1, 0.3, label='Phase 1', color=colors['Phase_1'])
    axes[1, 1].bar(bar_idx, s2, 0.3, bottom=s1, label='Phase 2', color=colors['Phase_2'])
    axes[1, 1].bar(bar_idx, s3, 0.3, bottom=[a + b for a, b in zip(s1, s2)], label='Phase 3', color=colors['Phase_3'])
    axes[1, 1].set_xticks(bar_idx)
    axes[1, 1].set_xticklabels([str(e) for e in sample_ep], fontsize=FONT_MIN)
    axes[1, 1].set_xlabel('Epoch', fontsize=FONT_AXIS)
    axes[1, 1].set_ylabel('Attention Weight', fontsize=FONT_AXIS)
    axes[1, 1].set_title('Error Decomposition Across Neighbourhoods', fontsize=FONT_MIN)
    axes[1, 1].legend(loc='upper left', fontsize=FONT_AXIS)
    axes[1, 1].tick_params(axis='both', labelsize=FONT_MIN)

    var_sample_idx = [min(i - variance_window, len(grad_variance_history['Phase_1']) - 1)
                      for i in sample_idx if i >= variance_window]
    if var_sample_idx:
        ax2 = axes[1, 1].twinx()
        for ph, col in colors.items():
            ax2.plot(bar_idx[:len(var_sample_idx)],
                     [grad_variance_history[ph][i] for i in var_sample_idx],
                     'o--', color=col, alpha=0.7, markersize=4)
        ax2.set_ylabel('Gradient Variance', fontsize=FONT_AXIS)
        ax2.set_yscale('log')
        ax2.tick_params(axis='both', labelsize=FONT_MIN)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    save_fig('plots/cl3e_gradient_analysis')

print("Gradient stability analysis plots saved to plots/")

print("ROC curve analysis completed and all visualizations saved.")