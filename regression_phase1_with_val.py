# regression_phase1_with_val.py
import argparse, datetime, os
from pathlib import Path

import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset_tw_multi import TWDatasetMulti

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c,out_c,3,padding=1),nn.BatchNorm2d(out_c),nn.ReLU(True),
            nn.Conv2d(out_c,out_c,3,padding=1),nn.BatchNorm2d(out_c),nn.ReLU(True))
    def forward(self,x): return self.net(x)

class UNetReg(nn.Module):
    def __init__(self, in_ch, base=32):
        super().__init__()
        self.enc1=DoubleConv(in_ch,base); self.pool1=nn.MaxPool2d(2)
        self.enc2=DoubleConv(base,base*2); self.pool2=nn.MaxPool2d(2)
        self.bott=DoubleConv(base*2,base*4)
        self.up2 = nn.ConvTranspose2d(base*4,base*2,2,2); self.dec2=DoubleConv(base*4,base*2)
        self.up1 = nn.ConvTranspose2d(base*2,base,2,2);   self.dec1=DoubleConv(base*2,base)
        self.out = nn.Conv2d(base,1,1)
    def forward(self,x):
        e1=self.enc1(x); e2=self.enc2(self.pool1(e1)); b=self.bott(self.pool2(e2))
        d2=self.dec2(torch.cat([self.up2(b),e2],1))
        d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        return x[:,0:1]+self.out(d1)

# 自訂 loss：對降雨 > 0.1 給較大權重
weighted_mse = lambda p,t: ((1+4*(t>0.1).float())*(p-t).pow(2)).mean()

def main_reg():
    pa=argparse.ArgumentParser(); add=pa.add_argument
    add("--rain_root",required=True); add("--t2m_root",required=True); add("--qpe_root",required=True)
    add("--topo",required=True); add("--mask",required=True)
    add("--years",nargs='+',type=int,default=[2021]); add("--epochs",type=int,default=30)
    add("--batch",type=int,default=8); add("--nhist",type=int,default=4); add("--patch",type=int,default=256)
    add("--lr",type=float,default=2e-4); add("--device",default="cuda"); add("--out",default="reg_multi.pth")
    add("--model_name",type=str,default="regM")
    args=pa.parse_args()

    ds = TWDatasetMulti(args.rain_root,args.t2m_root,args.qpe_root,args.topo,args.mask,
                        tuple(args.years),args.nhist,args.patch)
    train_len = int(len(ds) * 0.8)
    val_len = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])
    train_dl = DataLoader(train_ds,batch_size=args.batch,shuffle=True,num_workers=4,pin_memory=True)
    val_dl   = DataLoader(val_ds,batch_size=args.batch,shuffle=False,num_workers=4,pin_memory=True)

    in_ch = args.nhist*2 + 2
    model = UNetReg(in_ch).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    writer = SummaryWriter(Path("runs") / f"{args.model_name}_{datetime.datetime.now():%Y%m%d_%H%M%S}")

    gstep = 0
    for ep in range(1, args.epochs+1):
        model.train(); tot = 0
        for x,y in train_dl:
            x,y = x.to(args.device), y.to(args.device)
            opt.zero_grad(); loss = weighted_mse(model(x),y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tot += loss.item(); gstep += 1
            if gstep % 20 == 0:
                writer.add_scalar("loss/train_step", loss.item(), gstep)
        writer.add_scalar("loss/train_epoch", tot/len(train_dl), ep)

        model.eval(); vtot = 0
        with torch.no_grad():
            for x,y in val_dl:
                x,y = x.to(args.device), y.to(args.device)
                vtot += weighted_mse(model(x),y).item()
        writer.add_scalar("loss/val_epoch", vtot/len(val_dl), ep)

        sch.step()
        print(f"[Reg] ep{ep}/{args.epochs} train_loss={tot/len(train_dl):.4f} val_loss={vtot/len(val_dl):.4f}")

    torch.save(model.state_dict(), args.out)
    print("Model saved:", args.out)

if __name__ == "__main__":
    main_reg()
