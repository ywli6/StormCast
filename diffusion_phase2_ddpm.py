# ------------------------------------------------------------------------------------------
# diffusion_phase2_ddpm.py  —  StormCast Phase‑2 (簡化版, v1.1)
#  - 修正 Subset 無 nhist 屬性導致 AttributeError
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# diffusion_phase2_ddpm.py  —  StormCast Phase-2 (簡化版, v1.2)
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# diffusion_phase2_ddpm.py  —  StormCast Phase-2 (簡化版, v1.3)
# ------------------------------------------------------------------------------------------
import argparse, datetime
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter

from dataset_tw_multi import TWDatasetMulti
from regression_phase1_with_val import UNetReg   # Phase-1 regression

# ---------------- Residual Dataset --------------------------------------------------------
class ResidualDiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, reg_ckpt, device):
        self.base = base_ds
        root = base_ds.dataset if isinstance(base_ds, Subset) else base_ds
        in_ch = root.nhist * 2 + 2
        self.reg = UNetReg(in_ch).to(device)
        self.reg.load_state_dict(torch.load(reg_ckpt, map_location=device))
        self.reg.eval(); self.dev = device
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        with torch.no_grad():
            reg = self.reg(x.unsqueeze(0).to(self.dev))[0].cpu()
#        return torch.cat([x, reg], 0), y - reg          # (cond , residual)
        mask = x[-1:]
        res  = (y - reg) * mask
        return torch.cat([x, reg], 0), res          # (cond , residual)
# ---------------- UNet ε-predictor --------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, 3, 1, 1), nn.GroupNorm(8, oc), nn.SiLU(),
            nn.Conv2d(oc, oc, 3, 1, 1), nn.GroupNorm(8, oc), nn.SiLU())
    def forward(self,x): return self.net(x)

class UNetEps(nn.Module):
    def __init__(self, in_ch, base=32):
        super().__init__()
        self.e1=DoubleConv(in_ch,base); self.p1=nn.MaxPool2d(2)
        self.e2=DoubleConv(base,base*2); self.p2=nn.MaxPool2d(2)
        self.b = DoubleConv(base*2,base*4)
        self.u2=nn.ConvTranspose2d(base*4,base*2,2,2); self.d2=DoubleConv(base*4,base*2)
        self.u1=nn.ConvTranspose2d(base*2,base   ,2,2); self.d1=DoubleConv(base*2,base)
        self.out=nn.Conv2d(base,1,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.p1(e1)); b=self.b(self.p2(e2))
        d2=self.d2(torch.cat([self.u2(b),e2],1))
        d1=self.d1(torch.cat([self.u1(d2),e1],1))
        return self.out(d1)

# ---------------- DDPM utilities ----------------------------------------------------------
make_beta = lambda T: torch.linspace(1e-4, 2e-2, T)
def prep_a_bar(T, dev):
    beta = make_beta(T).to(dev)
    return torch.cumprod(1.-beta, 0)                  # ᾱ_t
q_sample = lambda x0,t,eps,a_bar: a_bar[t].sqrt().view(-1,1,1,1)*x0 + (1-a_bar[t]).sqrt().view(-1,1,1,1)*eps

# ---------------- Train -------------------------------------------------------------------
def main():
    pa=argparse.ArgumentParser(); add=pa.add_argument
    add('--rain_root',required=True); add('--t2m_root',required=True); add('--qpe_root',required=True)
    add('--topo',required=True); add('--mask',required=True); add('--reg_ckpt',required=True)
    add('--years',nargs='+',type=int,default=[2021]); add('--epochs',type=int,default=100)
    add('--batch',type=int,default=4); add('--nhist',type=int,default=4); add('--patch',type=int,default=256)
    add('--timesteps',type=int,default=1000); add('--device',default='cuda'); add('--out',default='diff_eps.pth')
    args=pa.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    base = TWDatasetMulti(args.rain_root,args.t2m_root,args.qpe_root,
                          args.topo,args.mask,tuple(args.years),args.nhist,args.patch)
    tr_len=int(len(base)*0.95)
    tr_ds,va_ds = random_split(base,[tr_len,len(base)-tr_len])
    tr_dl = DataLoader(ResidualDiffusionDataset(tr_ds,args.reg_ckpt,dev),batch_size=args.batch,shuffle=True ,num_workers=4)
    va_dl = DataLoader(ResidualDiffusionDataset(va_ds,args.reg_ckpt,dev),batch_size=args.batch,shuffle=False,num_workers=4)

    cond_ch = args.nhist*2 + 2 + 1   # X + reg_est
    net = UNetEps(cond_ch+1).to(dev)  # 再 +1 給 x_t
    opt = torch.optim.AdamW(net.parameters(),1e-4)
    a_bar=prep_a_bar(args.timesteps,dev)

    run_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(Path('runs')/f'diff_eps_{run_tag}')

    gstep=0
    for ep in range(1,args.epochs+1):
        # --- training ---
        net.train(); t_loss=0
        for cond,res in tr_dl:
            cond,res = cond.to(dev), res.to(dev)
            mask = cond[:, -2:-1]
            t = torch.randint(0,args.timesteps,(cond.size(0),),device=dev)
            eps = torch.randn_like(res)
            x_t = q_sample(res,t,eps,a_bar)
            pred = net(torch.cat([cond,x_t],1))
#           loss = nn.functional.mse_loss(pred,eps)
            loss = ((pred - eps).pow(2) * mask).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            t_loss += loss.item(); gstep+=1
            if gstep%50==0: writer.add_scalar('loss/train_step',loss.item(),gstep)

        # --- validation ---
        net.eval(); v_loss=0
        with torch.no_grad():
            for cond,res in va_dl:
                cond,res = cond.to(dev), res.to(dev)
                mask = cond[:, -2:-1]
                t = torch.randint(0,args.timesteps,(cond.size(0),),device=dev)
                eps = torch.randn_like(res)
                x_t = q_sample(res,t,eps,a_bar)
                pred = net(torch.cat([cond,x_t],1))
#                v_loss += nn.functional.mse_loss(pred,eps).item()
                v_loss += ((pred - eps).pow(2) * mask).mean().item()
        writer.add_scalars('loss/epoch',{'train':t_loss/len(tr_dl),'val':v_loss/len(va_dl)},ep)
        print(f'[DDPM] ep{ep}/{args.epochs} train={t_loss/len(tr_dl):.4f} val={v_loss/len(va_dl):.4f}')
        if ep%10==0:
            torch.save(net.state_dict(),Path(args.out).with_stem(f'diff_eps_ep{ep}'))

    torch.save(net.state_dict(),args.out)
    print(' saved', args.out)

if __name__ == '__main__':
    main()
