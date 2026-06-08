import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np

from preprocess_datasets import load_dataset, DATASET_CONFIGS
from models.model import CL3EGNN,  CL3E_ModelV2
from train import Trainer
from gradient_analysis import run_gradient_analysis
from utils import (
    plot_training_curves,
    plot_tsne,
    plot_tsne_three_phases,
    plot_curriculum_factors,
    plot_per_class_f1,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_multi_run_summary,
    plot_gradient_norm_history,
    plot_all_theory_validation,
    save_results,
    summarize_runs,
    print_summary_table,
)


ALL_DATASETS = ['cora', 'citeseer', 'pubmed', 'photo',
                'computers', 'cs', 'chameleon', 'ogbn-arxiv']


def get_args():
    parser = argparse.ArgumentParser(
        description='CL3E-GNN Experiments',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--dataset', type=str, default='computers',
        choices=ALL_DATASETS,
        help=(
            'Single dataset to run (default: cora).\n'
            'To run all datasets use --all_datasets.\n'
            'To run a custom subset use --datasets cora citeseer ...'
        ),
    )
    parser.add_argument(
        '--datasets', nargs='+', default=None,
        metavar='DS',
        help='Explicit list of datasets, e.g. --datasets cora citeseer pubmed',
    )
    parser.add_argument(
        '--all_datasets', action='store_true',
        help='Run all 8 benchmark datasets: ' + ', '.join(ALL_DATASETS),
    )
    parser.add_argument('--model',          type=str,   default='cl3e',
                        choices=['cl3e', 'v1', 'v2', 'v2orig'])
    parser.add_argument('--runs',           type=int,   default=1)
    parser.add_argument('--epochs',         type=int,   default=500)
    parser.add_argument('--hidden_dim',     type=int,   default=128)
    parser.add_argument('--num_gcn_layers', type=int,   default=3)
    parser.add_argument('--attention_type', type=str,   default='sigmoid',
                        choices=['sigmoid', 'softmax'])
    parser.add_argument('--dropout',        type=float, default=0.3)
    parser.add_argument('--lr',             type=float, default=0.005)
    parser.add_argument('--weight_decay',   type=float, default=5e-4)
    parser.add_argument('--grad_clip',      type=float, default=1.0)
    parser.add_argument('--dropedge',       type=float, default=0.3)
    parser.add_argument('--emb_reg',        type=float, default=1e-4)
    parser.add_argument('--patience',       type=int,   default=100)
    parser.add_argument('--log_interval',   type=int,   default=20)
    parser.add_argument('--loss_type',      type=str,   default='combined',
                        choices=['standard', 'curriculum', 'entropy', 'combined'])
    parser.add_argument('--progressive',    action='store_true',
                        help='Phase-1-Engage → Phase-2-Enact → Phase-3-Embed → Phase-4-Finetune')
    parser.add_argument('--dynamic_schedule', action='store_true',
                        help='Cycle standard → curriculum → entropy → combined')
    parser.add_argument('--data_root',      type=str,   default='./data')
    parser.add_argument('--out_dir',        type=str,   default='./results')
    parser.add_argument('--seed',           type=int,   default=42)
    args = parser.parse_args()

    # ── resolve final dataset list ────────────────────────────────────────────
    if args.all_datasets:
        args.datasets = ALL_DATASETS
    elif args.datasets:
        pass                        # explicit --datasets list wins
    else:
        args.datasets = [args.dataset]   # single --dataset (default: cora)

    return args


def build_model(args, in_dim, num_classes):
    if args.model == 'v2orig':
        return CL3E_ModelV2(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_layers=args.num_gcn_layers,
            dropout=args.dropout,
            attention_type=args.attention_type,
            use_batch_norm=True,
        )
    backbone = 0 if args.model == 'cl3e' else args.num_gcn_layers
    return CL3EGNN(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        backbone_layers=backbone,
        dropout=args.dropout,
        use_batch_norm=True,
        attention_type=args.attention_type,
    )


def _unpack_logits(out):
    if isinstance(out, (tuple, list)):
        if len(out) == 4:
            return out[1]
        if len(out) == 2:
            return out[0] if out[1].dim() == 1 else out[1]
    return out


def _collect_predictions(model, data, device, num_classes):
    """Returns y_true (np), y_pred (np), y_prob (np) on test nodes."""
    model.eval()
    with torch.no_grad():
        raw  = model(data.x.to(device), data.edge_index.to(device),
                     partial_labels=data.y.to(device),
                     mask=data.train_mask.to(device))
        out  = _unpack_logits(raw)
        mask = data.test_mask
        y_true = data.y[mask].cpu().numpy()
        y_pred = out[mask].argmax(1).cpu().numpy()
        y_prob = F.softmax(out[mask], dim=1).cpu().numpy()
    return y_true, y_pred, y_prob


def run_single(args, dataset_name, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    data, in_dim, num_classes = load_dataset(dataset_name, root=args.data_root)
    data = data.to(device)

    model = build_model(args, in_dim, num_classes).to(device)

    ckpt_dir = os.path.join(args.out_dir, 'checkpoints', dataset_name, f'seed{seed}')
    trainer  = Trainer(data, num_classes, device,
                       lr=args.lr, weight_decay=args.weight_decay,
                       grad_clip=args.grad_clip,
                       dropedge=args.dropedge,
                       emb_reg=args.emb_reg,
                       save_dir=ckpt_dir)

    curriculum_args = {
        'curriculum_type': 'performance',
        'initial_weight':  0.3,
        'final_weight':    1.0,
        'total_epochs':    args.epochs,
        'warmup_epochs':   max(20, args.epochs // 10),
        'smoothing':       0.15,
    }
    entropy_args = {
        'alpha':                0.2,
        'beta':                 0.05,
        'dynamic_weighting':    True,
        'confidence_threshold': 0.85,
    }

    if args.progressive and args.model == 'cl3e':
        model = trainer.train_progressive(
            model, epochs=args.epochs, log_interval=args.log_interval,
            curriculum_args=curriculum_args, entropy_args=entropy_args,
            patience=args.patience,
        )
    else:
        model = trainer.train(
            model, epochs=args.epochs, log_interval=args.log_interval,
            loss_type=args.loss_type,
            curriculum_args=curriculum_args, entropy_args=entropy_args,
            patience=args.patience, phase_tag=dataset_name,
            dynamic_schedule=args.dynamic_schedule,
        )

    metrics = trainer.evaluate(model)
    return metrics, trainer.history, model, data, num_classes


def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device  : {device}')
    print(f'Model   : {args.model}')
    print(f'Runs    : {args.runs}')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── generate static theory validation plots (no training data needed) ────
    theory_dir = os.path.join(args.out_dir, 'theory')
    os.makedirs(theory_dir, exist_ok=True)
    from utils import (plot_curriculum_schedule, plot_memory_forgetting_factor,
                       plot_imbalance_penalty, plot_curriculum_weight_evolution)
    plot_curriculum_schedule(
        save_path=os.path.join(theory_dir, 'thm3_curriculum_schedule.png'))
    plot_memory_forgetting_factor(
        save_path=os.path.join(theory_dir, 'cor4_memory_forgetting.png'))
    plot_imbalance_penalty(
        save_path=os.path.join(theory_dir, 'thm1_thm2_imbalance_penalty.png'))
    plot_curriculum_weight_evolution(
        save_path=os.path.join(theory_dir, 'thm3_weight_evolution.png'))
    print(f'Static theory plots saved to {theory_dir}/')

    all_summaries   = {}
    multi_run_store = {}

    for dataset_name in args.datasets:
        print(f'\n{"═"*70}')
        print(f'  Dataset: {dataset_name}')
        print(f'{"═"*70}')

        ds_dir = os.path.join(args.out_dir, dataset_name)
        os.makedirs(ds_dir, exist_ok=True)

        run_results  = []
        last_history = None
        last_model   = None
        last_data    = None
        last_nclass  = None

        for run in range(args.runs):
            seed = args.seed + run
            print(f'\n  Run {run+1}/{args.runs}  (seed={seed})')
            try:
                metrics, history, model, data, num_classes = run_single(
                    args, dataset_name, seed, device)
                run_results.append(metrics)
                last_history = history
                last_model   = model
                last_data    = data
                last_nclass  = num_classes
                print(f'  → ACC={metrics["acc"]:.4f}  '
                      f'AUC={metrics["auc"]:.4f}  '
                      f'F1-macro={metrics["f1_macro"]:.4f}')
            except Exception as e:
                print(f'  ✗ Run {run+1} failed: {e}')
                import traceback; traceback.print_exc()
                continue

        if not run_results:
            print(f'  All runs failed for {dataset_name}, skipping.')
            continue

        summary = summarize_runs(run_results)
        all_summaries[dataset_name]   = summary
        multi_run_store[dataset_name] = {
            'acc':      [r['acc']      for r in run_results],
            'auc':      [r['auc']      for r in run_results],
            'f1_macro': [r['f1_macro'] for r in run_results],
        }

        save_results(summary,     os.path.join(ds_dir, 'summary.json'))
        save_results(run_results, os.path.join(ds_dir, 'all_runs.json'))

        # ── per-dataset plots ─────────────────────────────────────────────────
        if last_history:
            plot_training_curves(
                last_history,
                save_path=os.path.join(ds_dir, 'training_curves.png'))

            from utils import plot_rademacher_proxy
            if last_history.get('train_acc') and last_history.get('val_acc'):
                plot_rademacher_proxy(
                    last_history['train_acc'], last_history['val_acc'],
                    save_path=os.path.join(ds_dir, 'thm2_rademacher_proxy.png'))

            if last_history.get('grad_norms'):
                plot_gradient_norm_history(
                    last_history['grad_norms'],
                    save_path=os.path.join(ds_dir, 'thm1_gradient_norms.png'))

        if last_model is not None and last_data is not None:
            y_true, y_pred, y_prob = _collect_predictions(
                last_model, last_data, device, last_nclass)

            # confusion matrix
            try:
                plot_confusion_matrix(
                    y_true, y_pred,
                    save_path=os.path.join(ds_dir, 'confusion_matrix.png'))
            except Exception as e:
                print(f'  confusion matrix failed: {e}')

            # per-class F1
            try:
                from sklearn.metrics import f1_score
                cf1 = f1_score(y_true, y_pred, average=None, zero_division=0)
                plot_per_class_f1(
                    cf1,
                    save_path=os.path.join(ds_dir, 'per_class_f1.png'))
            except Exception as e:
                print(f'  per-class F1 failed: {e}')

            # ROC curves
            try:
                plot_roc_curves(
                    y_true, y_prob, last_nclass,
                    save_path=os.path.join(ds_dir, 'roc_curves.png'))
            except Exception as e:
                print(f'  ROC curves failed: {e}')

            # t-SNE (global + per-class grid if >10 classes)
            try:
                plot_tsne(
                    last_model, last_data, device,
                    save_path=os.path.join(ds_dir, 'tsne.png'))
            except Exception as e:
                print(f'  t-SNE failed: {e}')

            # three-phase t-SNE progression (Engage → Enact → Embed)
            try:
                plot_tsne_three_phases(
                    last_model, last_data, device,
                    save_path=os.path.join(ds_dir, 'tsne_three_phases.png'),
                    dataset_name=dataset_name)
            except Exception as e:
                print(f'  Three-phase t-SNE failed: {e}')

            # curriculum factors
            try:
                if hasattr(last_model, 'curriculum_factor'):
                    plot_curriculum_factors(
                        last_model,
                        save_path=os.path.join(ds_dir, 'curriculum_factors.png'))
            except Exception as e:
                print(f'  curriculum factors failed: {e}')

            # gradient stability analysis (fresh model copy, all model types)
            try:
                grad_model = build_model(args, last_data.x.size(1), last_nclass)
                run_gradient_analysis(
                    grad_model, last_data, device,
                    out_dir=os.path.join(ds_dir, 'gradient_analysis'),
                    epochs=min(args.epochs, 200),
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    loss_type=args.loss_type,
                    N_train=int(last_data.train_mask.sum().item()),
                )
            except Exception as e:
                print(f'  Gradient analysis failed: {e}')

            # theory validation with actual training data
            try:
                plot_all_theory_validation(
                    last_history or {}, last_nclass,
                    model=last_model,
                    out_dir=ds_dir)
            except Exception as e:
                print(f'  theory plots failed: {e}')

    # ── cross-dataset summary plots ───────────────────────────────────────────
    print_summary_table(all_summaries)
    save_results(all_summaries, os.path.join(args.out_dir, 'all_summaries.json'))

    if multi_run_store:
        plot_multi_run_summary(
            multi_run_store,
            save_path=os.path.join(args.out_dir, 'multi_run_summary.png'))

    # ── model comparison: mean metrics per dataset as separate "models" ───────
    if all_summaries:
        comparison_input = {
            ds: {
                'acc':         s['acc']['mean'],
                'auc':         s['auc']['mean'],
                'f1_macro':    s['f1_macro']['mean'],
                'f1_weighted': s['f1_weighted']['mean'],
            }
            for ds, s in all_summaries.items()
        }
        plot_model_comparison(
            comparison_input,
            save_path=os.path.join(args.out_dir, 'model_comparison.png'))

    print(f'\nAll results saved to: {args.out_dir}')


if __name__ == '__main__':
    main()