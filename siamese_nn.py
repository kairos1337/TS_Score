import ast
import json
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import caerus_utils as cutils


class SiameseTimeSeriesDataset(Dataset):
    def __init__(self, bases, variants, scores, is_single_variant=False):
        self.is_single_variant = is_single_variant

        self.bases = [torch.tensor(b, dtype=torch.float32) for b in bases]
        self.variants = [[torch.tensor(v, dtype=torch.float32) for v in vs] for vs in variants]
        self.scores = torch.tensor(scores, dtype=torch.float32)

        self.pairs = []
        for i in range(len(bases)):
            for j in range(len(variants[i])):
                self.pairs.append((i, j))  # (base_index, variant_index)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        i, j = self.pairs[idx]
        return self.bases[i], self.variants[i][j], self.scores[i][j]

def siamese_collate(batch):
    b1, b2, y = zip(*batch)
    b1_pad = pad_sequence(b1, batch_first=True)
    b2_pad = pad_sequence(b2, batch_first=True)
    mask1 = torch.tensor([len(x) for x in b1])
    mask2 = torch.tensor([len(x) for x in b2])
    return b1_pad, b2_pad, mask1, mask2, torch.stack(y)

class TimeSeriesEncoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),nn.BatchNorm1d(32), nn.ReLU(),nn.Dropout(0.05),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.05),
            nn.Conv1d(64, hidden_dim, kernel_size=5, padding=2), nn.BatchNorm1d(hidden_dim), nn.ReLU(),nn.Dropout(0.05),

        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, lengths):
        x = x.unsqueeze(1)  # [B, 1, T]
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # [B, hidden_dim]
        return x

class SiameseNet(nn.Module):
    def __init__(self, encoder , hidden_dim=128):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(16, 1)
        )

    def forward(self, x1, x2, len1, len2):
        if torch.rand(1) < 0.5:
            x1, x2 = x2, x1
            len1, len2 = len2, len1

        # 2) encode both
        z1 = self.encoder(x1, len1)
        z2 = self.encoder(x2, len2)

        # 3) concatenate and head
        z = torch.cat([z1, z2], dim=1)
        return self.head(z).squeeze(1)

def train(model, dataloader, optimizer, device, epochs=10):
    model.train()
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for x1, x2, l1, l2, y in dataloader:
            x1, x2, l1, l2, y = x1.to(device), x2.to(device), l1.to(device), l2.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x1, x2, l1, l2)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(dataloader):.4f}")


def evaluate_with_metrics(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x1, x2, l1, l2, y in dataloader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            y_pred = model(x1, x2, l1.to(device), l2.to(device))
            all_preds.append(y_pred.cpu())
            all_targets.append(y.cpu())

    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    # Basic regression metrics
    mse = mean_squared_error(targets, preds)
    rmse = mse**0.5
    mae  = mean_absolute_error(targets, preds)
    r2   = r2_score(targets, preds)

    # Correlation metrics
    pearson_r, _  = pearsonr(targets, preds)
    spearman_r, _ = spearmanr(targets, preds)

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Pearson’s r": pearson_r,
        "Spearman’s ρ": spearman_r,
    }
    return metrics, preds, targets



def create_data_from_pickle():
    with open("synthetic_similarity_dataset.pkl", "rb") as f:
        data = pickle.load(f)

    bases = data["bases"][:1000]
    variants = data["variants"][:1000]
    scores = data["scores"][:1000]
    scores = np.array(scores)
    dataset = SiameseTimeSeriesDataset(bases, variants, scores,False)
    scores_np = dataset.scores.numpy()
    thresh = np.percentile(scores_np, 99)

    dataset.pairs = [
        (i, j)
        for (i, j) in dataset.pairs
        if scores_np[i, j] <= thresh
    ]
    return dataset


def create_data_from_df(df):
    """
    df must have columns:
      - 'head_norm_y'  : strings like "[0.1, 0.2, …]"
      - 'ocr_norm_y'   : same shape
      - 'visual_score' : float in [0,1]
    Returns three lists: bases, variants, scores

    """
    bases, variants, scores, variants2 = [], [], [], []
    df['head_norm_y'] = df['head_norm_y'].apply(json.loads)
    df['ocr_norm_y'] = df['ocr_norm_y'].apply(json.loads)
    bases = [val for val in df['head_norm_y']]
    scores = [[value] for value in df['visual_score']]
    scores = np.array(scores)
    for i in range(len(bases)):
        variants.append([df['ocr_norm_y'][i]])

    dataset = SiameseTimeSeriesDataset(bases, variants, scores)
    return dataset


def plot_test_results():
    plt.figure(figsize=(6, 6))
    plt.scatter(true_vals, pred_vals, alpha=0.4)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("True Score")
    plt.ylabel("Predicted Score")
    plt.title("Validation Predictions vs True Scores")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



conn = cutils.DbConnection()
conn.connect()
df = conn.get_ocrs_and_scores()
dataset= create_data_from_df(df)
#dataset = create_data_from_pickle()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


total_len = len(dataset)
val_size = int(0.05 * total_len)
train_size = total_len - val_size
train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=1028, shuffle=True, collate_fn=siamese_collate, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=1028, shuffle=False, collate_fn=siamese_collate,    num_workers=4,pin_memory=True)

encoder = TimeSeriesEncoder(hidden_dim=128)
model = SiameseNet(encoder, hidden_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

train(model, train_loader, optimizer, device, epochs=5)
metrics, pred_vals, true_vals = evaluate_with_metrics(model, val_loader, device)

for name, val in metrics.items():
    print(f"{name:12s}: {val:.4f}")

plot_test_results()
