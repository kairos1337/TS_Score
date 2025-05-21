import ast
import json
import pickle

import numpy as np
import torch
from dtaidistance import dtw
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
from ploting import score_dis, plot_test_results, plot_graphs
from sklearn.preprocessing import MinMaxScaler


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

def siamese_collate(batch):
    b1, b2, y = zip(*batch)
    b1_pad = pad_sequence(b1, batch_first=True)
    b2_pad = pad_sequence(b2, batch_first=True)
    mask1 = torch.tensor([len(x) for x in b1])
    mask2 = torch.tensor([len(x) for x in b2])
    return b1_pad, b2_pad, mask1, mask2, torch.stack(y)

def train(dataset_for_train, epochs=10, path = "my_sweet_model.pth"):

    train_loader, val_loader, model, optimizer, scheduler, hidden_dim = set_up_nn(dataset_for_train)
    model.train()
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for x1, x2, l1, l2, y in train_loader:
            x1, x2, l1, l2, y = x1.to(device), x2.to(device), l1.to(device), l2.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x1, x2, l1, l2)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

    torch.save({
        'hidden_dim': hidden_dim,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # …
    }, path)
    print("Model saved to", path)

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
    """
    Create dataset from Alex ocrs and head
    :return:
    """
    with open("synthetic_similarity_dataset2.pkl", "rb") as f:
        data = pickle.load(f)

    bases = data["bases"]
    variants = data["variants"]
    scores = data["scores"]
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
def dtw_distance(ts1: np.ndarray, ts2: np.ndarray, normalize: bool = True) -> float:
    dist = dtw.distance_fast(ts1.astype(float), ts2.astype(float), penalty=0.1)
    return dist / max(len(ts1), len(ts2)) if normalize else dist

def adding_dtw(df):
    df = df.copy()
    scaler = MinMaxScaler()

    def compute_row_dtw(row):
        # parse
        head = np.array(json.loads(row['head_norm_y']), dtype=float)
        ocr  = np.array(json.loads(row['ocr_norm_y']), dtype=float)
        # normalize in-place for DTW
        all_vals = np.concatenate([head, ocr]).reshape(-1, 1)

        scaler = MinMaxScaler().fit(all_vals)
        head_scaled = scaler.transform(head.reshape(-1, 1)).ravel()
        ocr_scaled = scaler.transform(ocr.reshape(-1, 1)).ravel()
        # distance
        return 1 - dtw_distance(head_scaled, ocr_scaled, normalize=True)

    df['dtw_dist'] = df.apply(compute_row_dtw, axis=1)
    return df

def set_up_nn(dataset_for_train, hidden_dim=128):
    total_len = len(dataset_for_train)
    val_size = int(0.05 * total_len)
    train_size = total_len - val_size
    train_set, val_set = random_split(dataset_for_train, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, collate_fn=siamese_collate, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1028, shuffle=False, collate_fn=siamese_collate, num_workers=4,
                            pin_memory=True)
    encoder = TimeSeriesEncoder(hidden_dim=hidden_dim)
    model = SiameseNet(encoder, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    return train_loader, val_loader, model, optimizer, scheduler, hidden_dim

def predict_pair(ts1, ts2, model):
    # ts1, ts2: 1-D numpy arrays or Python lists
    x1 = torch.tensor(ts1, dtype=torch.float32).to(device).unsqueeze(0)  # [1, T1]
    x2 = torch.tensor(ts2, dtype=torch.float32).to(device).unsqueeze(0)  # [1, T2]
    len1 = torch.tensor([x1.size(1)], dtype=torch.long).to(device)
    len2 = torch.tensor([x2.size(1)], dtype=torch.long).to(device)
    with torch.no_grad():
        score = model(x1, x2, len1, len2)
    return score.item()

def load_model(path):
    ckpt = torch.load(path, map_location='cpu')
    hidden_dim = ckpt['hidden_dim']
    encoder = TimeSeriesEncoder(hidden_dim=hidden_dim)
    model = SiameseNet(encoder, hidden_dim=hidden_dim)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    return model

def normalize_series_list(json_str, feature_range=(0, 1)):
    # 1) parse JSON → numpy array
    arr = np.array(json.loads(json_str), dtype=float)
    # 2) avoid divide-by-zero on constant series
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return json.dumps(arr.tolist())
    # 3) scale to [0,1] (or any feature_range)
    scaled = (arr - mn) / (mx - mn)
    # 4) back to JSON (or return as Python list if you prefer)
    return json.dumps(scaled.tolist())

def test_model_on_real_data():
    """
    This function load the ocrs from database, load a model and make prediction
    Later the function show the results with plotly
    :return:
    """

    df_real_data = conn.get_ocrs_and_scores()
    for col in ['head_norm_y', 'ocr_norm_y']:
        df_real_data[col] = df_real_data[col].apply(normalize_series_list)

    df_real_data['visual_score'] = df_real_data['visual_score'] / 100.0
    model = load_model(path)
    preds = []
    for idx, row in df_real_data.iterrows():
        # if strings like "[0.1,0.2,...]" use json.loads; otherwise skip
        ts1 = json.loads(row["head_norm_y"]) if isinstance(row["head_norm_y"], str) else row["head_norm_y"]
        ts2 = json.loads(row["ocr_norm_y"]) if isinstance(row["ocr_norm_y"], str) else row["ocr_norm_y"]

        preds.append(predict_pair(ts1, ts2, model))

    df_real_data["predicted_similarity"] = preds
    df_real_data.to_csv("with_predictions.csv", index=False)
    plot_test_results(df_real_data['visual_score'].values, preds)

    #plot_graphs(df_real_data, "predicted_similarity")



device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
path= "my_sweet_model.pth"


conn = cutils.DbConnection()
conn.connect()
df = conn.get_ocrs_and_scores()
score_dis(df)
#test_model_on_real_data()
#dataset_for_train = create_data_from_pickle()

#train(dataset_for_train)

#test_model_on_real_data()


#for name, val in metrics.items():
#    print(f"{name:12s}: {val:.4f}")

#plot_test_results(true_vals, pred_vals)
