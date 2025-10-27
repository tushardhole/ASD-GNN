import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from graph_dataset import FCGraphDataset

# -----------------------------
# Configuration
# -----------------------------
FC_DIR = "./testdata/fc_matrices"
CSV_FILE = "./testdata/labels.csv"
WHICH_FC = "combined"  # 'fisher', 'partial', or 'combined'
TOP_K = 10
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Dataset
# -----------------------------
dataset = FCGraphDataset(CSV_FILE, FC_DIR, which_fc=WHICH_FC, top_k=TOP_K)
print(f"✅ Loaded {len(dataset)} samples, each graph has {dataset.get(0).x.shape[0]} nodes.")

# Train/val split
indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=dataset.df['label'])
train_set = torch.utils.data.Subset(dataset, train_idx)
val_set = torch.utils.data.Subset(dataset, val_idx)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# -----------------------------
# Model
# -----------------------------
class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

model = SimpleGCN(in_channels=dataset.get(0).x.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# -----------------------------
# Training
# -----------------------------
best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    total_loss /= len(train_loader.dataset)

    # Validate
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
    val_acc = correct / len(val_loader.dataset)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
    print(f"Epoch {epoch:02d}, Loss: {total_loss:.4f}, Val Acc: {val_acc:.4f}")

print(f"✅ Best Validation Accuracy: {best_val_acc:.4f}")
