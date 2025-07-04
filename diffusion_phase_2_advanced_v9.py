"""
Diffusion Phase‑2 (v9e) — full NaN‑safe script
=============================================
完全覆蓋檔案。可直接存成 `diffusion_phase_2_advanced_v9e.py` 執行。
"""
import argparse, math, datetime, contextlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

# ─── 使用者資料集 / Phase-1 回歸 ────────────────────────────────────────────
from dataset_tw_multi import TWDatasetMulti
from regression_phase1_with_val import UNetReg

# ─── Karras σ schedule + time embed ─────────────────────────────────────────
def karras_sigma(n: int, smin: float, smax: float, rho: float = 5.0, device="cpu"):
    t = torch.linspace(0, 1, n, device=device)
    inv = 1.0 / rho
    return (smax ** inv + t * (smin ** inv - smax ** inv)) ** rho

def t_embed(log_sigma: torch.Tensor, dim: int = 256):
    half = dim // 2
    freqs = torch.exp(-math.log(1e4) * torch.arange(half, device=log_sigma.device) / half)
    emb = log_sigma[:, None] * freqs[None]
    return torch.cat([torch.sin(emb), torch.cos(emb)], 1)

# ─── Residual Dataset ───────────────────────────────────────────────────────
class ResidualDS(torch.utils.data.Dataset):
    def __init__(self, base: TWDatasetMulti, reg_ckpt: str, device: torch.device):
        self.b = base; self.dev = device; eps = 1e-3
        self.nhist = base.nhist
        # 轉成 Tensor、保證非零
        self.y_m = torch.tensor(float(base.norm['y_m']))
        self.y_s = torch.tensor(float(abs(base.norm['y_s']))).clamp_min(eps)
        self.reg = UNetReg(self.nhist*2+2).to(device)
        self.reg.load_state_dict(torch.load(reg_ckpt, map_location=device)); self.reg.eval()

    def _lz(self, r):
        return (torch.log1p(r.clamp_min(0)) - self.y_m) / self.y_s

    def __len__(self): return len(self.b)

    def __getitem__(self, idx):
        x, y = self.b[idx]          # x:C×H×W, y:1×H×W
        _, H, W = x.shape
        pad_h, pad_w = (4-H%4)%4, (4-W%4)%4
        x_pad = F.pad(x, (0,pad_w,0,pad_h))
        with torch.no_grad():
            reg_pad = self.reg(x_pad.unsqueeze(0).to(self.dev))[0].cpu()
        reg = reg_pad[..., :H, :W]
        cond = torch.nan_to_num(torch.cat([x, reg], 0))
        res = torch.nan_to_num((self._lz(y) - self._lz(reg)).clamp(-10,10))
        return cond, res

# ─── UNet ε predictor ───────────────────────────────────────────────────────
class ResBlk(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__(); self.fc = nn.Linear(emb, cout)
        self.c1 = nn.Conv2d(cin, cout, 3,1,1); self.c2 = nn.Conv2d(cout, cout,3,1,1)
        self.skip = nn.Conv2d(cin, cout,1) if cin!=cout else nn.Identity(); self.act=nn.SiLU()
    def forward(self,x,e):
        h=self.act(self.c1(x)+self.fc(e)[:,:,None,None]); h=self.act(self.c2(h)); return h+self.skip(x)

class UNetEPS(nn.Module):
    def __init__(self, ch_in, base=64, emb=256):
        super().__init__(); self.tp=nn.Sequential(nn.Linear(emb,emb*4),nn.SiLU(),nn.Linear(emb*4,emb))
        self.cin=nn.Conv2d(ch_in,base,3,1,1); self.d1=ResBlk(base,base*2,emb); self.p1=nn.MaxPool2d(2)
        self.d2=ResBlk(base*2,base*4,emb); self.p2=nn.MaxPool2d(2); self.mid=ResBlk(base*4,base*4,emb)
        self.u2=ResBlk(base*4+base*2,base*2,emb); self.u1=ResBlk(base*2+base,base,emb); self.out=nn.Conv2d(base,1,1)
    def forward(self,x,s):
        e=self.tp(t_embed(s.log())); h1=self.cin(x); h2=self.d1(self.p1(h1),e); h3=self.d2(self.p2(h2),e)
        m=self.mid(h3,e); u2=self.u2(torch.cat([F.interpolate(m, h2.shape[-2:],mode='nearest'),h2],1),e)
        u1=self.u1(torch.cat([F.interpolate(u2, h1.shape[-2:],mode='nearest'),h1],1),e)
        return self.out(u1)

# ─── EDM loss ───────────────────────────────────────────────────────────────
def edm(pred, eps, s, s_data=0.5):
    w=(s**2)/(s**2+s_data**2); return (w.view(-1,1,1,1)*(pred-eps).square()).mean()

# ─── Training ───────────────────────────────────────────────────────────────
def train(cfg):
    dev=torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    tr_b=TWDatasetMulti(cfg.rain_root,cfg.t2m_root,cfg.qpe_root,cfg.topo,cfg.mask,tuple(cfg.train_years),cfg.nhist,cfg.patch or None)
    va_b=TWDatasetMulti(cfg.rain_root,cfg.t2m_root,cfg.qpe_root,cfg.topo,cfg.mask,tuple(cfg.val_years),cfg.nhist,cfg.patch or None)
    tr_ds,va_ds=ResidualDS(tr_b,cfg.reg_ckpt,dev),ResidualDS(va_b,cfg.reg_ckpt,dev)
    tr_dl=DataLoader(tr_ds,batch_size=cfg.batch,shuffle=True,num_workers=4,pin_memory=True)
    va_dl=DataLoader(va_ds,batch_size=cfg.batch,shuffle=False,num_workers=4,pin_memory=True)

    in_ch=cfg.nhist*2+3+1; net=UNetEPS(in_ch).to(dev); ema=UNetEPS(in_ch).to(dev); ema.load_state_dict(net.state_dict())
    opt=torch.optim.AdamW(net.parameters(),lr=cfg.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg.epochs*max(len(tr_dl),1))
    scaler=GradScaler(enabled=not cfg.no_amp)
    sig=karras_sigma(cfg.steps,cfg.sigma_min,cfg.sigma_max,device=dev)

    out=Path(cfg.out); out.mkdir(parents=True,exist_ok=True); tb=SummaryWriter(out/'tb')
    gs=0
    for ep in range(1,cfg.epochs+1):
        net.train(); tl=0
        for cond,res in tr_dl:
            cond,res=cond.to(dev),res.to(dev); B=cond.size(0)
            idx=torch.randint(0,cfg.steps,(B,),device=dev); s=sig[idx]
            eps=torch.randn_like(res); x=res+s.view(-1,1,1,1)*eps
            ctx=autocast(enabled=not cfg.no_amp,device_type='cuda') if torch.cuda.is_available() else contextlib.nullcontext()
            with ctx:
                pred=net(torch.cat([cond,x],1),s); loss=edm(pred,eps,s)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(),0.5)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); tl+=loss.item(); gs+=1
            # EMA
            with torch.no_grad():
                for pe,p in zip(ema.parameters(),net.parameters()): pe.mul_(0.9999).add_(p,alpha=1-0.9999)
            if gs%50==0: tb.add_scalar('train/loss_step',loss.item(),gs)
        tb.add_scalar('train/loss_epoch',tl/len(tr_dl),ep)


        # val
        net.eval(); vl=0; n=0
        with torch.no_grad():
            for cond,res in va_dl:
                cond,res=cond.to(dev),res.to(dev); B=cond.size(0)
                idx=torch.randint(0,cfg.steps,(B,),device=dev); s=sig[idx]
                eps=torch.randn_like(res); x=res+s.view(-1,1,1,1)*eps
                pred=net(torch.cat([cond,x],1),s)
                vl+=edm(pred,eps,s).item()*B; n+=B
        vl/=max(n,1); tb.add_scalar('val/loss_epoch',vl,ep)
        print(f"Ep {ep:03d} | train {tl/len(tr_dl):.4f} | val {vl:.4f}")
        if ep%cfg.ckpt_int==0 or ep==cfg.epochs:
            torch.save({'model':net.state_dict(),'ema':ema.state_dict()}, out/f'model_ep{ep}.pth')

    tb.close()

# ─── CLI ────────────────────────────────────────────────────────────────────
if __name__=='__main__':
    pa=argparse.ArgumentParser(); a=pa.add_argument
    a('--rain_root'); a('--t2m_root'); a('--qpe_root'); a('--topo'); a('--mask'); a('--reg_ckpt')
    a('--train_years',nargs='+',type=int,default=[2021]); a('--val_years',nargs='+',type=int,default=[2022])
    a('--nhist',type=int,default=4); a('--patch',type=int,default=0)
    a('--epochs',type=int,default=10); a('--batch',type=int,default=4); a('--lr',type=float,default=3e-4)
    a('--device',default='cuda'); a('--sigma_min',type=float,default=0.02); a('--sigma_max',type=float,default=5.0)
    a('--steps',type=int,default=1000); a('--ckpt_int',type=int,default=5); a('--no_amp',action='store_true')
    a('--out',default='outputs/diff_p2_v9e')
    cfg=pa.parse_args(); train(cfg)
