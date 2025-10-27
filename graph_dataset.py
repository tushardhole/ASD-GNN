import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import dense_to_sparse

class FCGraphDataset(Dataset):
    """
    Graph dataset for fMRI FC matrices (Fisher and Partial).
    Pads and sparsifies graph for GNN input.
    """

    def __init__(self, csv_file, fc_dir, which_fc='combined', top_k=10, transform=None):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.fc_dir = fc_dir
        self.which_fc = which_fc
        self.top_k = int(top_k)
        self.transform = transform

        # Ensure CSV has required columns
        if 'file_name' not in self.df.columns:
            raise ValueError("CSV must contain 'file_name'")
        if 'DX_GROUP' not in self.df.columns:
            raise ValueError("CSV must contain 'DX_GROUP'")

        # Encode labels
        self.df['label'] = LabelEncoder().fit_transform(self.df['DX_GROUP'].astype(int))

    def len(self):
        return len(self.df)

    def _load_fc(self, base):
        """Load FC matrices and combine as requested."""
        f_fisher = os.path.join(self.fc_dir, f"{base}_fisher.npy")
        f_partial = os.path.join(self.fc_dir, f"{base}_partial.npy")

        if not os.path.exists(f_fisher):
            raise FileNotFoundError(f"Missing FC matrix: {f_fisher}")

        fc_fisher = np.load(f_fisher).astype(np.float32)
        fc_fisher = np.nan_to_num(fc_fisher)

        if self.which_fc == 'fisher':
            fc = fc_fisher
        elif self.which_fc == 'partial':
            if os.path.exists(f_partial):
                fc_partial = np.load(f_partial).astype(np.float32)
                fc_partial = np.nan_to_num(fc_partial)
                fc = fc_partial
            else:
                fc = fc_fisher
        elif self.which_fc == 'combined':
            if os.path.exists(f_partial):
                fc_partial = np.load(f_partial).astype(np.float32)
                fc_partial = np.nan_to_num(fc_partial)
                fc = 0.5 * (fc_fisher + fc_partial)
            else:
                fc = fc_fisher
        else:
            raise ValueError(f"Unknown which_fc mode: {self.which_fc}")

        # Symmetrize & zero diagonal
        fc = 0.5 * (fc + fc.T)
        np.fill_diagonal(fc, 0.0)
        fc = np.nan_to_num(fc)
        return fc

    def get(self, idx):
        row = self.df.iloc[idx]
        base = row['file_name'].replace('.nii.gz', '').replace('.nii', '')
        label = int(row['label'])

        fc = self._load_fc(base)
        N = fc.shape[0]

        # Top-k sparsification
        if self.top_k < (N - 1):
            kth = max(1, min(self.top_k, N - 1))
            partitioned = np.argpartition(np.abs(fc), -kth, axis=1)[:, -kth:]
            mask = np.zeros_like(fc, dtype=bool)
            rows = np.repeat(np.arange(N), kth)
            cols = partitioned.ravel()
            mask[rows, cols] = True
            mask = mask | mask.T
            sparse_fc = np.zeros_like(fc)
            sparse_fc[mask] = fc[mask]
        else:
            sparse_fc = fc.copy()

        # Dense -> Sparse
        t_sparse = torch.tensor(sparse_fc)
        edge_index, edge_attr = dense_to_sparse(t_sparse)
        edge_attr = edge_attr.unsqueeze(1).float()

        # Node features
        node_mean = fc.mean(axis=1)
        node_std = fc.std(axis=1)
        node_degree = (np.abs(sparse_fc) > 0).sum(axis=1).astype(np.float32)
        node_strength = np.abs(sparse_fc).sum(axis=1)
        node_clustering = (fc @ fc).diagonal()

        x = np.stack([node_mean, node_std, node_degree, node_strength, node_clustering], axis=1).astype(np.float32)
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
        x = torch.tensor(x, dtype=torch.float)

        y = torch.tensor(label, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index.long(), edge_attr=edge_attr, y=y)
        if self.transform:
            data = self.transform(data)
        return data
