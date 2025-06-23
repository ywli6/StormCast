import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# ---- Dataset Class ----
class QPesumTreadDataset(Dataset):
    def __init__(self, year, train=True, root_dir="D:/2025/NCDR AI/Data"):
        self.root_dir = root_dir
        self.year = year
        # Build list of valid dates
        self.date_list = [f"{year}{m:02d}{d:02d}" for m in range(1, 13) for d in range(1, 32)
                          if os.path.exists(os.path.join(root_dir, "tread/rain", f"ERA5_WRF.rain.Taiwan.0.01deg.{year}{m:02d}{d:02d}.nc"))]
        # Paths
        self.rain_paths = [os.path.join(root_dir, "tread/rain", f"ERA5_WRF.rain.Taiwan.0.01deg.{dt}.nc") for dt in self.date_list]
        self.t2m_paths  = [os.path.join(root_dir, "tread/t2m",  f"ERA5_WRF.t2m.Taiwan.0.01deg.{dt}.nc") for dt in self.date_list]
        self.qpe_paths  = [os.path.join(root_dir, "qpesum",     f"cwb_QPESUM_rain_hourly-grid_0.01deg-{dt}.nc") for dt in self.date_list]
        # Constants
        topo_ds = xr.open_dataset(os.path.join(root_dir, "constants", "TWmap-0.01deg-GIS.nc"))
        mask_ds = xr.open_dataset(os.path.join(root_dir, "constants", "land_sea_mask.nc"))
        self.topo = topo_ds['topo'].values.astype(np.float32)
        self.mask = mask_ds['mask'].values.astype(np.float32)
        topo_ds.close(); mask_ds.close()
        # Lat/Lon
        ds0 = xr.open_dataset(self.rain_paths[0])
        self.lat = ds0['lat'].values
        self.lon = ds0['lon'].values
        ds0.close()
        self.transform = None

    def __len__(self):
        return len(self.date_list)

    def __getitem__(self, idx):
        # Load data
        ds_r = xr.open_dataset(self.rain_paths[idx]); rain = ds_r['rain'].values.astype(np.float32); ds_r.close()
        ds_t = xr.open_dataset(self.t2m_paths[idx]); t2m  = ds_t['t2m'].values.astype(np.float32); ds_t.close()
        ds_q = xr.open_dataset(self.qpe_paths[idx]); qpe  = ds_q['rain'].values.astype(np.float32); ds_q.close()
        # Shape (T, H, W)
        T, H, W = rain.shape
        # Repeat topo/mask
        topo = np.repeat(self.topo[np.newaxis,:,:], T, axis=0)
        mask = np.repeat(self.mask[np.newaxis,:,:], T, axis=0)
        # Stack channels: (4, T, H, W)
        x = np.stack([rain, t2m, topo, mask], axis=0)
        y = qpe[np.newaxis,:,:]
        # To tensor
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        if self.transform:
            x, y = self.transform(x, y)
        return x, y

# ---- Normalization ----
def compute_norm_params(dataset, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    ys = []
    for x, y in loader:
        y = torch.nan_to_num(y, nan=0.0)
        y_log = torch.log1p(y)
        ys.append(y_log.view(y_log.size(0), -1))
    all_y = torch.cat(ys, dim=0)
    mean = all_y.mean().item()
    std = all_y.std().item()
    print(f"→ Y log1p mean={mean:.4f}, std={std:.4f}")
    return {'qpe': {'mean': mean, 'std': std}}

# ---- Model ----
class RegressionNet(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ---- Training ----
def train_regression(model, loader, optimizer, device, norm_params, epochs=10):
    writer = SummaryWriter('runs/regression')
    criterion = nn.MSELoss()
    model.to(device).train()
    losses = []
    for ep in range(epochs):
        total_loss = 0.0
        for x, y in loader:
            # x: (B,4,T,H,W), y: (B,1,T,H,W)
            B, C, T, H, W = x.shape
            x = x.permute(0,2,1,3,4).reshape(B*T, C, H, W).to(device)
            y = y.permute(0,2,1,3,4).reshape(B*T, 1, H, W).to(device)
            y = torch.nan_to_num(y, nan=0.0)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * (B*T)
        avg = total_loss / (len(loader.dataset) * T)
        losses.append(avg)
        writer.add_scalar('Loss/train', avg, ep)
        print(f"[Epoch {ep+1}/{epochs}] Loss: {avg:.4f}")
    torch.save(model.state_dict(), 'reg_net.pth')
    print(" Model saved to reg_net.pth")
    plt.figure()
    plt.plot(losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig('loss_curve.png')
    writer.close()

# ---- Evaluation ----
def evaluate_regression(model, loader, device, norm_params, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    model.to(device).eval()
    total_mae = total_mse = total_pixels = 0
    details = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            B, C, T, H, W = x.shape
            x = x.permute(0,2,1,3,4).reshape(B*T, C, H, W).to(device)
            y = y.permute(0,2,1,3,4).reshape(B*T, 1, H, W).to(device)
            y = torch.nan_to_num(y, nan=0.0)
            pred = model(x)
            yp = torch.expm1(pred * norm_params['qpe']['std'] + norm_params['qpe']['mean'])
            yt = torch.expm1(y * norm_params['qpe']['std'] + norm_params['qpe']['mean'])
            abs_err = (yp - yt).abs()
            sq_err = (yp - yt)**2
            total_mae += abs_err.sum().item()
            total_mse += sq_err.sum().item()
            total_pixels += yt.numel()
            for i in range(min(2, B*T)):
                mae_s = abs_err[i].mean().item()
                rmse_s = torch.sqrt(sq_err[i].mean()).item()
                details.append({'sample': f'{idx}_{i}', 'MAE': mae_s, 'RMSE': rmse_s})
                plt.imsave(os.path.join(out_dir, f'pred_{idx}_{i}.png'), yp[i,0].cpu(), cmap='Blues')
                plt.imsave(os.path.join(out_dir, f'true_{idx}_{i}.png'), yt[i,0].cpu(), cmap='Greens')
    pd.DataFrame(details).to_csv(os.path.join(out_dir, 'evaluation_details.csv'), index=False)
    mae = total_mae / total_pixels
    rmse = (total_mse / total_pixels)**0.5
    print(f" Test MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# ---- Main ----
if __name__ == '__main__':
    # Datasets
    train_ds = QPesumTreadDataset(2021, train=True)
    test_ds  = QPesumTreadDataset(2022, train=False)
    # Norm params
    norm_params = compute_norm_params(train_ds)
    # Transform: normalize y only
    def transform(x, y):
        return x, torch.nan_to_num((torch.log1p(y) - norm_params['qpe']['mean']) / norm_params['qpe']['std'], nan=0.0)
    train_ds.transform = transform
    test_ds.transform  = transform
    # Loaders
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False, num_workers=0)
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(' Using device:', device)
    # Model & optimizer
    net = RegressionNet(in_ch=4).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    # Train & Eval
    train_regression(net, train_loader, optimizer, device, norm_params, epochs=10)
    evaluate_regression(net, test_loader, device, norm_params)
