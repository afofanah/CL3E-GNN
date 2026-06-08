import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize

from models.loss import get_loss, CurriculumLoss, EntropyRegularizedLoss, CombinedLoss


def _dropedge(edge_index, drop_rate: float, training: bool):
    if not training or drop_rate <= 0.0:
        return edge_index
    keep = torch.rand(edge_index.size(1), device=edge_index.device) > drop_rate
    return edge_index[:, keep]


def _torch_load(path, device):
    """torch.load with weights_only=True to silence FutureWarning."""
    return torch.load(path, map_location=device, weights_only=True)


class Trainer:
    def __init__(self, data, num_classes, device,
                 lr=0.005, weight_decay=5e-4,
                 grad_clip=1.0, dropedge=0.3, emb_reg=1e-4,
                 save_dir='./checkpoints'):
        self.data         = data
        self.num_classes  = num_classes
        self.device       = device
        self.lr           = lr
        self.weight_decay = weight_decay
        self.grad_clip    = grad_clip
        self.dropedge     = dropedge
        self.emb_reg      = emb_reg
        self.save_dir     = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.history = {k: [] for k in
                        ['train_loss', 'val_loss', 'train_acc',
                         'val_acc', 'test_acc', 'f1']}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _forward(self, model, training=False):
        ei  = _dropedge(self.data.edge_index, self.dropedge, training)
        out = model(self.data.x, ei,
                    partial_labels=self.data.y, mask=self.data.train_mask)

        if isinstance(out, (tuple, list)):
            if len(out) == 4:
                # CL3EClassifier: (embeddings, logits, aux_engage, aux_enact)
                return out[0], out[1], out[2], out[3]
            if len(out) == 2:
                # Distinguish by shape:
                #   CL3EClassifier/EmbedModule: out[0]=(N,H), out[1]=(N,C) → emb, logits
                #   CL3E_ModelV1/V2:            out[0]=(N,C), out[1]=(H,)  → logits, graph_repr
                a, b = out[0], out[1]
                if b.dim() == 1:
                    # b is a 1-D graph-level summary → a is the node logits
                    return None, a, None, None
                else:
                    # both 2-D → standard (embeddings, logits)
                    return a, b, None, None
        # bare logits tensor
        return None, out, None, None

    def _acc(self, pred, mask):
        return pred[mask].eq(self.data.y[mask]).float().mean().item()

    def _update_criterion(self, criterion, outputs, targets):
        if isinstance(criterion, (CurriculumLoss, CombinedLoss)):
            criterion.update_class_performance(F.softmax(outputs, dim=1), targets)
        elif isinstance(criterion, EntropyRegularizedLoss):
            criterion.update_weights(outputs, targets)

    def _make_scheduler(self, optimizer, total_epochs, warmup_epochs=10):
        """Linear warmup then cosine annealing, called once per epoch."""
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(warmup_epochs, 1))
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _train_step(self, model, optimizer, criterion, epoch):
        """Single training epoch. Scheduler stepped separately."""
        model.train()
        optimizer.zero_grad()

        emb, out, logits_engage, logits_enact = self._forward(model, training=True)
        tl  = self.data.y[self.data.train_mask].long()
        to_ = out[self.data.train_mask]

        # primary loss
        loss = criterion(to_, tl)

        # auxiliary phase supervision — delayed until epoch 10
        if logits_engage is not None and logits_enact is not None:
            if hasattr(model, 'aux_loss') and epoch >= 10:
                loss = loss + model.aux_loss(
                    logits_engage[self.data.train_mask],
                    logits_enact[self.data.train_mask],
                    tl
                )

        # embedding L2 regularisation
        if emb is not None and self.emb_reg > 0:
            loss = loss + self.emb_reg * emb.pow(2).mean()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        optimizer.step()

        if hasattr(criterion, 'update_epoch'):
            criterion.update_epoch(epoch)
        self._update_criterion(criterion, to_.detach(), tl)
        return loss.item()

    def _eval_logits(self, model):
        """Pure inference — always restores training mode afterwards."""
        was_training = model.training
        model.eval()
        with torch.no_grad():
            _, out, _, _ = self._forward(model, training=False)
        if was_training:
            model.train()
        return out

    def _record_metrics(self, model, criterion, loss_val):
        """Eval + append to history. Leaves model in eval mode."""
        model.eval()
        with torch.no_grad():
            _, out, _, _ = self._forward(model, training=False)
            import torch.nn as _nn
            _pure_ce = _nn.CrossEntropyLoss()
            vl    = _pure_ce(out[self.data.val_mask],
                             self.data.y[self.data.val_mask].long()).item()
            pred  = out.argmax(1)
            tacc  = self._acc(pred, self.data.train_mask)
            vacc  = self._acc(pred, self.data.val_mask)
            tsacc = self._acc(pred, self.data.test_mask)
            y_ts = self.data.y[self.data.test_mask].cpu().numpy()
            p_ts = pred[self.data.test_mask].cpu().numpy()
            f1   = (f1_score(y_ts, p_ts, average='weighted', zero_division=0)
                    if len(y_ts) > 0 else 0.0)
        self.history['train_loss'].append(loss_val)
        self.history['val_loss'].append(vl)
        self.history['train_acc'].append(tacc)
        self.history['val_acc'].append(vacc)
        self.history['test_acc'].append(tsacc)
        self.history['f1'].append(f1)
        return vacc, tacc, tsacc

    # ── standard training ─────────────────────────────────────────────────────

    def train(self, model, epochs=500, log_interval=10, loss_type='combined',
              curriculum_args=None, entropy_args=None, patience=50,
              phase_tag='full', dynamic_schedule=False):

        model     = model.to(self.device)
        optimizer = optim.AdamW(model.parameters(),
                                lr=self.lr, weight_decay=self.weight_decay)
        scheduler = self._make_scheduler(optimizer, epochs, warmup_epochs=10)

        if dynamic_schedule:
            return self._train_dynamic_schedule(
                model, optimizer, scheduler, epochs, log_interval,
                curriculum_args, entropy_args, patience, phase_tag)

        criterion  = get_loss(loss_type, self.num_classes, curriculum_args, entropy_args)
        best_val   = 0.0
        no_improve = 0
        ckpt       = os.path.join(self.save_dir, f'{phase_tag}_best.pt')

        for epoch in range(epochs):
            loss_val = self._train_step(model, optimizer, criterion, epoch)

            # lightweight val check every epoch (no history write)
            out  = self._eval_logits(model)
            vacc = self._acc(out.argmax(1), self.data.val_mask)

            if vacc > best_val:
                best_val   = vacc
                no_improve = 0
                torch.save(model.state_dict(), ckpt)
            else:
                no_improve += 1

            # step scheduler once per epoch
            scheduler.step()

            # full metrics + print at log_interval
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                vacc_full, tacc, tsacc = self._record_metrics(model, criterion, loss_val)
                print(f'[{phase_tag}] Ep {epoch+1:04d} | loss {loss_val:.4f} '
                      f'| train {tacc:.4f} | val {vacc_full:.4f} | test {tsacc:.4f}')

            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1} '
                      f'(best val {best_val:.4f})')
                break

        if os.path.exists(ckpt):
            model.load_state_dict(_torch_load(ckpt, self.device))
        return model

    # ── progressive three-phase training ─────────────────────────────────────

    def train_progressive(self, model, epochs=500, log_interval=10,
                          curriculum_args=None, entropy_args=None, patience=50):
        phase_epochs = epochs // 4

        phase_defs = [
            ('Phase-1-Engage',   lambda m: list(m.engage.parameters())),
            ('Phase-2-Enact',    lambda m: list(m.enact.parameters())),
            ('Phase-3-Embed',    lambda m: list(m.embed.parameters())),
            ('Phase-4-Finetune', lambda m: list(m.parameters())),
        ]

        for phase_name, get_params in phase_defs:
            print(f'\n{"─"*60}\n  {phase_name}\n{"─"*60}')
            for p in model.parameters():
                p.requires_grad = False
            for p in get_params(model):
                p.requires_grad = True

            lr_val    = self.lr if 'Finetune' not in phase_name else self.lr / 5
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr_val, weight_decay=self.weight_decay)
            scheduler = self._make_scheduler(optimizer, phase_epochs, warmup_epochs=5)
            criterion = get_loss('combined', self.num_classes,
                                 curriculum_args, entropy_args)
            ckpt       = os.path.join(self.save_dir, f'{phase_name}_best.pt')
            best_val   = 0.0
            no_improve = 0

            for epoch in range(phase_epochs):
                loss_val = self._train_step(model, optimizer, criterion, epoch)

                out  = self._eval_logits(model)
                vacc = self._acc(out.argmax(1), self.data.val_mask)

                if vacc > best_val:
                    best_val   = vacc
                    no_improve = 0
                    torch.save(model.state_dict(), ckpt)
                else:
                    no_improve += 1

                scheduler.step()

                if (epoch + 1) % log_interval == 0 or epoch == 0:
                    vacc_full, tacc, tsacc = self._record_metrics(
                        model, criterion, loss_val)
                    print(f'  [{phase_name}] Ep {epoch+1:03d} | loss {loss_val:.4f} '
                          f'| train {tacc:.4f} | val {vacc_full:.4f} | test {tsacc:.4f}')

                if no_improve >= patience:
                    print(f'  Early stopping at epoch {epoch+1}')
                    break

            if os.path.exists(ckpt):
                model.load_state_dict(_torch_load(ckpt, self.device))

        for p in model.parameters():
            p.requires_grad = True
        return model

    # ── dynamic loss schedule ─────────────────────────────────────────────────

    def _train_dynamic_schedule(self, model, optimizer, scheduler, epochs,
                                log_interval, curriculum_args, entropy_args,
                                patience, phase_tag):
        loss_types = ['standard', 'curriculum', 'entropy', 'combined']
        seg_epochs = epochs // len(loss_types)
        best_val   = 0.0
        no_improve = 0
        ckpt       = os.path.join(self.save_dir, f'{phase_tag}_best.pt')

        for loss_type in loss_types:
            print(f'\n  Dynamic schedule → {loss_type} loss ({seg_epochs} epochs)')
            criterion = get_loss(loss_type, self.num_classes,
                                 curriculum_args, entropy_args)

            for epoch in range(seg_epochs):
                loss_val = self._train_step(model, optimizer, criterion, epoch)

                out  = self._eval_logits(model)
                vacc = self._acc(out.argmax(1), self.data.val_mask)

                if vacc > best_val:
                    best_val   = vacc
                    no_improve = 0
                    torch.save(model.state_dict(), ckpt)
                else:
                    no_improve += 1

                scheduler.step()

                if (epoch + 1) % log_interval == 0 or epoch == 0:
                    vacc_full, tacc, tsacc = self._record_metrics(
                        model, criterion, loss_val)
                    print(f'  [{loss_type}] Ep {epoch+1:03d} | loss {loss_val:.4f} '
                          f'| train {tacc:.4f} | val {vacc_full:.4f}')

                if no_improve >= patience:
                    print(f'  Early stopping [{loss_type}] at epoch {epoch+1}')
                    break

        if os.path.exists(ckpt):
            model.load_state_dict(_torch_load(ckpt, self.device))
        return model

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, model):
        model.eval()
        with torch.no_grad():
            _, out, _, _ = self._forward(model, training=False)
            mask   = self.data.test_mask
            y_true = self.data.y[mask].cpu().numpy()
            y_pred = out[mask].argmax(1).cpu().numpy()
            y_prob = F.softmax(out[mask], dim=1).cpu().numpy()

        acc    = accuracy_score(y_true, y_pred)
        f1_mac = f1_score(y_true, y_pred, average='macro',    zero_division=0)
        f1_wei = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        try:
            if self.num_classes == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                yb  = label_binarize(y_true, classes=list(range(self.num_classes)))
                auc = roc_auc_score(yb, y_prob, multi_class='ovr', average='macro')
        except Exception:
            auc = float('nan')

        return {'acc': acc, 'auc': auc, 'f1_macro': f1_mac, 'f1_weighted': f1_wei}