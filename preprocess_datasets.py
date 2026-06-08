import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikipediaNetwork
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset


DATASET_CONFIGS = {
    'cora':      {'nodes': 2708,   'edges': 5278,    'classes': 7,  'features': 1433},
    'citeseer':  {'nodes': 3327,   'edges': 4552,    'classes': 6,  'features': 3703},
    'photo':     {'nodes': 7650,   'edges': 119081,  'classes': 8,  'features': 745},
    'cs':        {'nodes': 18333,  'edges': 163788,  'classes': 15, 'features': 6805},
    'computers': {'nodes': 13752,  'edges': 491722,  'classes': 10, 'features': 767},
    'pubmed':    {'nodes': 19717,  'edges': 88648,   'classes': 3,  'features': 5414},
    'chameleon': {'nodes': 2277,   'edges': 36101,   'classes': 5,  'features': 2325},
    'ogbn-arxiv':{'nodes': 169343, 'edges': 1166243, 'classes': 40, 'features': 128},
}


def load_dataset(name: str, root: str = './data'):
    name = name.lower()
    transform = T.NormalizeFeatures()

    if name in ('cora', 'citeseer', 'pubmed'):
        dataset = Planetoid(root=f'{root}/{name}', name=name, transform=transform)
        data = dataset[0]

    elif name == 'photo':
        dataset = Amazon(root=f'{root}/photo', name='Photo', transform=transform)
        data = dataset[0]
        data = _add_masks(data)

    elif name == 'computers':
        dataset = Amazon(root=f'{root}/computers', name='Computers', transform=transform)
        data = dataset[0]
        data = _add_masks(data)

    elif name == 'cs':
        dataset = Coauthor(root=f'{root}/cs', name='CS', transform=transform)
        data = dataset[0]
        data = _add_masks(data)

    elif name == 'chameleon':
        dataset = WikipediaNetwork(root=f'{root}/chameleon', name='chameleon', transform=transform)
        data = dataset[0]
        if not hasattr(data, 'train_mask') or data.train_mask.dim() > 1:
            data = _add_masks(data)

    elif name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=f'{root}/ogbn-arxiv',
                                         transform=T.ToUndirected())
        data = dataset[0]
        data.y = data.y.squeeze(1)
        split_idx = dataset.get_idx_split()
        n = data.num_nodes
        data.train_mask = _idx_to_mask(split_idx['train'], n)
        data.val_mask   = _idx_to_mask(split_idx['valid'], n)
        data.test_mask  = _idx_to_mask(split_idx['test'],  n)

    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASET_CONFIGS.keys())}")

    num_classes = int(data.y.max().item()) + 1
    num_features = data.x.size(1)
    return data, num_features, num_classes


def _idx_to_mask(idx, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = True
    return mask


def _add_masks(data, train_ratio=0.6, val_ratio=0.2, seed=42):
    torch.manual_seed(seed)
    n = data.num_nodes
    perm = torch.randperm(n)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    data.train_mask = _idx_to_mask(perm[:train_end], n)
    data.val_mask   = _idx_to_mask(perm[train_end:val_end], n)
    data.test_mask  = _idx_to_mask(perm[val_end:], n)
    return data