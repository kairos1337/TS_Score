import pickle

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

class SiameseTimeSeriesDataset(Dataset):
    def __init__(self, bases, variants, scores):
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
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, hidden_dim, kernel_size=5, padding=2), nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, lengths):
        x = x.unsqueeze(1)  # [B, 1, T]
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # [B, hidden_dim]
        return x

class SiameseNet(nn.Module):
    def __init__(self, encoder , hidden_dim=64):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x1, x2, len1, len2):
        z1 = self.encoder(x1, len1)
        z2 = self.encoder(x2, len2)
        z = torch.cat([z1, z2], dim=1)
        return self.head(z).squeeze(1)  # output shape: [B]

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




with open("synthetic_similarity_dataset.pkl", "rb") as f:
    data = pickle.load(f)

bases    = data["bases"]      # np.ndarray בגודל (10000, T)
variants = data["variants"]   # רשימת רשימות של np.ndarray באורכים משתנים
scores   = data["scores"]     # np.ndarray בגודל (10000, 5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = SiameseTimeSeriesDataset(bases, variants, scores)


total_len = len(dataset)
val_size = int(0.1 * total_len)
train_size = total_len - val_size
train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=siamese_collate)
val_loader   = DataLoader(val_set, batch_size=64, shuffle=False, collate_fn=siamese_collate)

encoder = TimeSeriesEncoder(hidden_dim=64)
model = SiameseNet(encoder, hidden_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train(model, train_loader, optimizer, device, epochs=20)
metrics, pred_vals, true_vals = evaluate_with_metrics(model, val_loader, device)
for name, val in metrics.items():
    print(f"{name:12s}: {val:.4f}")
plt.figure(figsize=(6, 6))
plt.scatter(true_vals, pred_vals, alpha=0.4)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("True Score")
plt.ylabel("Predicted Score")
plt.title("Validation Predictions vs True Scores")
plt.grid(True)
plt.tight_layout()
plt.show()
