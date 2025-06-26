# ------------------------------------------------------------------------------------------
# inference_diffusion_single_day_v4.py  (steps param + dual padding + squeeze fix)
# ------------------------------------------------------------------------------------------
import argparse
from pathlib import Path
import torch, xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature

from dataset_tw_multi import TWDatasetMulti
from regression_phase1_with_val import UNetReg
from diffusion_phase2_ddpm import UNetEps

# ---------------- DDPM sampling (sub‑steps) ----------------------------------------------
make_beta = lambda T, dev: torch.linspace(1e-4, 2e-2, T, device=dev)

@torch.no_grad()
def ddpm_sample(cond, net, T=1000, steps=200, device="cpu"):
    idx = torch.linspace(0, T-1, steps, dtype=torch.long, device=device)
    betas  = make_beta(T, device)[idx]
    alphas = 1 - betas; a_bar = torch.cumprod(alphas, 0)
    x = torch.randn((1,1,*cond.shape[-2:]), device=device)
    for i in reversed(range(steps)):
        eps = net(torch.cat([cond, x], 1))
        coef1 = 1/alphas[i].sqrt(); coef2 = (1-alphas[i])/(1-a_bar[i]).sqrt()
        mean = coef1*(x - coef2*eps)
        x = mean if i==0 else mean + betas[i].sqrt()*torch.randn_like(x)
    return x  # (1,1,H,W)

# ---------------- utils ------------------------------------------------------------------

def pad4(t):
    _,_,h,w=t.shape; ph=(4-h%4)%4; pw=(4-w%4)%4
    return torch.nn.functional.pad(t,(0,pw,0,ph)), ph, pw

CMAP,VMIN,VMAX="Blues",0,10

def plot_panel(ax,lon,lat,data,title):
    ax.set_extent([lon.min(),lon.max(),lat.min(),lat.max()],ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE,linewidth=.5); ax.add_feature(cfeature.BORDERS,linestyle=":")
    m=ax.pcolormesh(lon,lat,data,cmap=CMAP,vmin=VMIN,vmax=VMAX,shading="auto",transform=ccrs.PlateCarree())
    ax.set_title(title,fontsize=8); plt.colorbar(m,ax=ax,orientation="vertical",shrink=.7,pad=.02)

# ---------------- main -------------------------------------------------------------------

def main():
    pa=argparse.ArgumentParser(); add=pa.add_argument
    add("--date",required=True); add("--steps",type=int,default=200)
    add("--rain_root",required=True); add("--t2m_root",required=True); add("--qpe_root",required=True)
    add("--topo",required=True); add("--mask",required=True)
    add("--reg_ckpt",default="reg_multi.pth"); add("--diff_ckpt",default="diff_eps.pth")
    add("--nhist",type=int,default=4); add("--out_dir",default="./outputs"); add("--device",default="cuda")
    args=pa.parse_args(); dev=torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds=TWDatasetMulti(args.rain_root,args.t2m_root,args.qpe_root,
                      args.topo,args.mask,(int(args.date[:4]),),args.nhist,
                      patch=0,random_crop=False,save_norm=False)
    base=ds.dates.index(args.date)*24; norm=ds.norm

    in_c=args.nhist*2+2
    reg=UNetReg(in_c).to(dev); reg.load_state_dict(torch.load(args.reg_ckpt,map_location=dev)); reg.eval()
    diff=UNetEps(in_c+2).to(dev); diff.load_state_dict(torch.load(args.diff_ckpt,map_location=dev)); diff.eval()

    r_nc=xr.open_dataset(ds.rain_map[args.date]); coarse=r_nc["rain"].load(); lat,lon=r_nc["lat"].values,r_nc["lon"].values
    Path(args.out_dir).mkdir(parents=True,exist_ok=True)

    den=lambda z: torch.expm1(z.detach()*norm["y_s"]+norm["y_m"]).cpu().numpy().squeeze()

    for hr in range(24):
        x,y=ds[base+hr]
        x_pad,ph,pw=pad4(x.unsqueeze(0).to(dev)); reg_pad=reg(x_pad)[0].cpu()
        reg_est=reg_pad[..., :-ph or None, :-pw or None]

        cond=torch.cat([x,reg_est],0).unsqueeze(0).to(dev)
        cond_pad,ph2,pw2=pad4(cond); res_pad=ddpm_sample(cond_pad,diff,T=1000,steps=args.steps,device=dev)[0].cpu()
        res_hat=res_pad[..., :-ph2 or None, :-pw2 or None]

        reg_hr,true_hr,pred_hr=map(den,[reg_est,y,reg_est+res_hat])

        fig,axs=plt.subplots(2,2,figsize=(10,9),subplot_kw={'projection':ccrs.PlateCarree()})
        ts=f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]} {hr:02d}Z"
        plot_panel(axs[0,0],lon,lat,coarse.isel(time=hr),f"TReAD Rain • {ts}")
        plot_panel(axs[0,1],lon,lat,reg_hr,f"Regression Est • {ts}")
        plot_panel(axs[1,0],lon,lat,true_hr,f"QPESUM • {ts}")
        plot_panel(axs[1,1],lon,lat,pred_hr,f"Diffusion Pred • {ts}")
        plt.tight_layout(); fname=Path(args.out_dir,f"{args.date}_T{hr:02d}.png"); plt.savefig(fname,dpi=180); plt.close(); print("saved",fname)

if __name__=='__main__':
    main()
