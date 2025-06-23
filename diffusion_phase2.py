# diffusion_phase2.py
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../physicsnemo"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from physicsnemo.models.diffusion.unet import UNet as PhysicsNeMoUNet
from regression_phase1 import QPesumTreadDataset, RegressionNet
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torchvision


# 1. 二階段資料集：高解析背景 + 迴歸預估 → 預測殘差
class ResidualDiffusionDataset(Dataset):
    def __init__(self, base_ds, reg_model, device):
        self.base_ds = base_ds
        self.reg_model = reg_model.to(device).eval()
        self.device = device
        self.per_day = self.base_ds[0][0].shape[1] - 1  # 時間長度 T - 1

    def __len__(self):
        return len(self.base_ds) * self.per_day

    def __getitem__(self, idx):
        day = idx // self.per_day
        t = idx % self.per_day + 1
        x, y = self.base_ds[day]

        hr_bg = y[:, t-1]
        x_t = x[:, t].unsqueeze(0).to(self.device)
        with torch.no_grad():
            reg_est = self.reg_model(x_t)
        reg_est = reg_est.squeeze(0).cpu()

        residual = y[:, t] - reg_est
        return hr_bg, reg_est, residual


# 2. 擴散模型：UNet 接收 hr_bg 和 reg_est，輸出殘差
class ResidualDiffusionNet(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.unet = PhysicsNeMoUNet(
            img_resolution=resolution,
            img_in_channels=2,
            img_out_channels=1
        )

    def forward(self, hr_bg, reg_est):
        x = torch.cat([hr_bg, reg_est], dim=1)
        return self.unet(x)


# 3. 訓練流程
def train_diffusion(model, loader, optimizer, device, num_epochs=10, writer=None):
    criterion = nn.MSELoss()
    model.to(device).train()

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        model.train()

        for hr_bg, reg_est, residual in loader:
            hr_bg = hr_bg.to(device)
            reg_est = reg_est.to(device)
            tgt_res = residual.to(device)

            pred_res = model(hr_bg, reg_est)
            loss = criterion(pred_res, tgt_res)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * hr_bg.size(0)

        avg_loss = total_loss / len(loader.dataset)
        print(f"[Diffusion Epoch {epoch}/{num_epochs}] Loss: {avg_loss:.6f}")

        if writer:
            writer.add_scalar("Loss/train", avg_loss, epoch)

        if epoch % 5 == 0 and writer:
            model.eval()
            with torch.no_grad():
                hr_bg, reg_est, residual = next(iter(loader))
                hr_bg = hr_bg.to(device)
                reg_est = reg_est.to(device)
                residual = residual.to(device)

                pred_res = model(hr_bg, reg_est)
                vis = torch.cat([hr_bg, reg_est, residual, pred_res], dim=0)
                vis_grid = torchvision.utils.make_grid(vis, nrow=hr_bg.size(0))
                writer.add_image("Prediction/sample", vis_grid, epoch)


# 4. 主程式入口
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_ds = QPesumTreadDataset(year=2012, train=True)
    reg_model = RegressionNet()
    reg_model.load_state_dict(torch.load("regression_model.pt", map_location=device))

    ds = ResidualDiffusionDataset(base_ds, reg_model, device)
    train_loader = DataLoader(ds, batch_size=2, shuffle=True)

    diffusion_net = ResidualDiffusionNet(resolution=ds[0][0].shape[-2:])
    optimizer = optim.Adam(diffusion_net.parameters(), lr=1e-4)

    writer = SummaryWriter("runs/diffusion_phase2")

    train_diffusion(diffusion_net, train_loader, optimizer, device, num_epochs=20, writer=writer)

    torch.save(diffusion_net.state_dict(), "diffusion_model.pt")
