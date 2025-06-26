# =========================  StormCast Taiwan – Multi‑Var Upgrade  =========================
# 此 Canvas 版本整合「雨量 (rain) + 溫度 (t2m) + 地形 + 陸海掩碼」四大輸入，
# 並區分資料夾：
#   • rain_root →  D:\2025\NCDR AI\Data\tread\rain
#   • t2m_root  →  D:\2025\NCDR AI\Data\tread\t2m
#   • topo_path →  D:\2025\NCDR AI\Data\constants\TWmap-0.01deg-GIS.nc
#   • mask_path →  D:\2025\NCDR AI\Data\constants\land_sea_mask.nc
#   • qpe_root  →  D:\2025\NCDR AI\Data\qpesum
# 主要檔案：
#   1) dataset_tw_multi.py          ▶︎ 升級 Dataset (rain+t2m+topo+mask)
#   2) regression_phase1_upgrade.py ▶︎ U‑Net 迴歸（多變數版）
#   3) diffusion_phase2_ddpm.py     ▶︎ DDPM 殘差擴散（多變數版）
#   4) inference_diffusion_single_day_fixed.py (保持不變，僅需把 in_ch 更新)
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# 1) dataset_tw_multi.py
# ------------------------------------------------------------------------------------------
#"""Dataset for Taiwan 0.01°: rain + t2m + topo + land‑sea mask
#* rain, t2m : daily NetCDF (24×721×681)
#* topo, mask: constant NetCDF
#Channel order：
#   [rain_hist0 … rain_hist{Nh-1},
#    t2m_hist0  … t2m_hist{Nh-1},
#    topo, mask]
#變數名稱：
#   rain → "rain"   (mm/hr)  (log1p)
#   t2m  → "t2m"    (°C)     (z‑score)
#   topo → "elevation" (m)    (z‑score)
#   mask → <第一個 2D 變數>     (0/1)
#"""

#"""Dataset for Taiwan 0.01°: rain + t2m + topo + land‑sea mask
#* rain, t2m : daily NetCDF (24×lat×lon) — time dim 可能叫 `time` 或 `hourly`
#* topo, mask: constant NetCDF
#Channel order： rain_hist • t2m_hist • topo • mask
#"""

# =======================  dataset_tw_multi.py  (v1.4)  =======================
# Taiwan 0.01° Dataset: rain (mm/h), t2m (°C), topo (m), land-sea mask (0/1)
# Channel order : [ rain_hist*N , t2m_hist*N , topo , mask ]
# ---------------------------------------------------------------------------
# =======================  dataset_tw_multi.py  (v1.6)  =======================
# Taiwan 0.01° Dataset: rain (mm/h), t2m (°C), topo (m), land‑sea mask (0/1)
# 本版修正：
#   • 將 QPESUM 填充值 ±1e9 視為缺測並轉 NaN，再以 np.nan_to_num→0 處理
#   • _stat_log() 加上 isnan 檢查，確保 μ/σ 永不為 NaN
#   • 其餘結構與 v1.4 相同
#-----------------------------------------------------------------------------
# =======================  dataset_tw_multi.py  (v1.7)  =======================
# Taiwan 0.01° Dataset: rain (mm/h), t2m (°C), topo (m), land‑sea mask (0/1)
# 亮點（修復版）：
#   • 將 QPESUM 等 NetCDF 以 ±1e9 當缺值的格點設為 NaN，再歸零
#   • _stat_log() 防 NaN / σ=0
#   • 語法完整，Pylance 不再報錯
#---------------------------------------------------------------------------
import glob, json, os, random, re, math
from typing import Tuple, List

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

CLIP_RAIN       = (0.0, 20.0)   # mm/h clip before log1p
DEFAULT_PATCH   = 256
FILL_THRESHOLD  = 9e8            # |value| > 9e8 視為填充值

# ------------------------ helper ------------------------
_date_pat = re.compile(r"(\d{8})(?=\.nc$)")

def _extract_date(path: str) -> str:
    m = _date_pat.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse date from {path}")
    return m.group(1)

def open_nc(path: str):
    ds = xr.open_dataset(path)
    # 有些檔把時間維度命名為 hourly
    if "hourly" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"hourly": "time"})
    return ds

# 填值(±1e9) → np.nan
_fill_nan = lambda arr: np.where(np.abs(arr) > FILL_THRESHOLD, np.nan, arr)

def _get_norm_path(root: str):
    return os.path.join(root, "norm_params.json")

# ------------------------ Dataset ------------------------
class TWDatasetMulti(Dataset):
    def __init__(self,
                 rain_root: str,
                 t2m_root : str,
                 qpe_root : str,
                 topo_path: str,
                 mask_path: str,
                 years    : Tuple[int, ...] = (2021,),
                 nhist    : int = 4,
                 patch    : int = DEFAULT_PATCH,
                 random_crop: bool = True,
                 save_norm: bool = True):

        self.nhist, self.patch, self.random_crop = nhist, patch, random_crop

        # ---- collect files ----
        def ls(root): return sorted(glob.glob(os.path.join(root, "*.nc")))
        rain_files, t2m_files, qpe_files = map(ls, (rain_root, t2m_root, qpe_root))
        yrs = {str(y) for y in years}
        rain_files = [f for f in rain_files if _extract_date(f)[:4] in yrs]
        t2m_files  = [f for f in t2m_files  if _extract_date(f)[:4] in yrs]
        qpe_files  = [f for f in qpe_files  if _extract_date(f)[:4] in yrs]
        assert len(rain_files)==len(t2m_files)==len(qpe_files), "file count mismatch"

        self.dates    = [_extract_date(f) for f in rain_files]
        self.rain_map = {_extract_date(f): f for f in rain_files}
        self.t2m_map  = {_extract_date(f): f for f in t2m_files}
        self.qpe_map  = {_extract_date(f): f for f in qpe_files}

        # topo / mask  (constant 2‑D)
        self.topo = xr.open_dataset(topo_path)[list(xr.open_dataset(topo_path).data_vars.keys())[0]].values.astype("float32")
        self.mask = xr.open_dataset(mask_path)[list(xr.open_dataset(mask_path).data_vars.keys())[0]].values.astype("float32")

        # ---- normalization ----
        self.norm_path = _get_norm_path(rain_root)
        if os.path.exists(self.norm_path):
            with open(self.norm_path) as f: self.norm = json.load(f)
        else:
            self.norm = self._scan_norm(sample=200)
            if save_norm:
                with open(self.norm_path, "w") as f: json.dump(self.norm, f, indent=2)

    # --------------------- utils -------------------------
    @staticmethod
    def _stat_log(arr):
        arr  = _fill_nan(arr)
        arr  = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr  = np.clip(arr, *CLIP_RAIN)
        arr  = np.log1p(arr)
        mean = float(np.nanmean(arr)) if not math.isnan(np.nanmean(arr)) else 0.0
        std  = float(np.nanstd(arr))
        if math.isnan(std) or std < 1e-6:
            std = 1e-6
        return mean, std

    def _scan_norm(self, sample=200):
        print("[Dataset] Scanning mean/std …")
        rains, t2ms, ys = [], [], []
        for d in self.dates[:sample]:
            rains.append(_fill_nan(open_nc(self.rain_map[d])["rain"].values[:self.nhist]))
            t2ms .append(_fill_nan(open_nc(self.t2m_map[d]) ["t2m" ].values[:self.nhist]))
            ys   .append(_fill_nan(open_nc(self.qpe_map[d])  ["rain"].values))
        rains = np.concatenate(rains,0)
        t2ms  = np.concatenate(t2ms ,0)
        ys    = np.concatenate(ys   ,0)

        rain_m, rain_s = self._stat_log(rains)
        y_m,    y_s    = self._stat_log(ys)
        t2m_m,  t2m_s  = self._stat_log(t2ms)
        topo_m, topo_s = float(np.nanmean(self.topo)), max(float(np.nanstd(self.topo)),1e-6)

        return {"rain_m": rain_m, "rain_s": rain_s,
                "y_m": y_m,       "y_s": y_s,
                "t2m_m": t2m_m,  "t2m_s": t2m_s,
                "topo_m": topo_m, "topo_s": topo_s}

    # -------------------- Dataset API --------------------
    def __len__(self):
        return len(self.dates)*24

    def __getitem__(self, idx):
        date_idx, hr = divmod(idx, 24)
        date = self.dates[date_idx]

        rain_ds = open_nc(self.rain_map[date])
        t2m_ds  = open_nc(self.t2m_map [date])
        qpe_ds  = open_nc(self.qpe_map [date])

        # ---- history stacks ----
        rain_hist, t2m_hist = [], []
        for h in range(hr-self.nhist+1, hr+1):
            hh = max(h,0)
            rain_hist.append(_fill_nan(rain_ds["rain"].isel(time=hh).values))
            t2m_hist .append(_fill_nan(t2m_ds ["t2m" ].isel(time=hh).values))
        rain_arr = np.nan_to_num(np.stack(rain_hist,0).astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)
        t2m_arr  = np.nan_to_num(np.stack(t2m_hist ,0).astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)
        y_raw    = _fill_nan(qpe_ds["rain"].isel(time=hr).values.astype("float32"))
        y_arr    = np.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- crop (optional) ----
        H,W = y_arr.shape
        if self.patch:
            ph = self.patch
            top  = random.randint(0,H-ph) if self.random_crop else (H-ph)//2
            left = random.randint(0,W-ph) if self.random_crop else (W-ph)//2
            sl, sc = slice(top,top+ph), slice(left,left+ph)
            rain_arr = rain_arr[:,sl,sc]
            t2m_arr  = t2m_arr [:,sl,sc]
            y_arr    = y_arr[sl,sc]
            topo = self.topo[sl,sc]; mask = self.mask[sl,sc]
        else:
            topo, mask = self.topo, self.mask

        # ---- normalization ----
        n = self.norm
        rain_arr = (np.log1p(np.clip(rain_arr,*CLIP_RAIN)) - n["rain_m"]) / n["rain_s"]
        t2m_arr  = (t2m_arr - n["t2m_m"]) / n["t2m_s"]
        topo     = (topo    - n["topo_m"]) / n["topo_s"]
        y_norm   = (np.log1p(np.clip(y_arr,*CLIP_RAIN)) - n["y_m"]) / n["y_s"]

        x = np.concatenate([rain_arr, t2m_arr, topo[None,:,:], mask[None,:,:]], 0)
        return torch.from_numpy(x), torch.from_numpy(y_norm).unsqueeze(0)

# ------------------- external helper -------------------
def load_norm(rain_root: str):
    with open(_get_norm_path(rain_root)) as f:
        return json.load(f)
# ===========================================================================
