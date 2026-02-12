#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# RRTMG_example.py

# Clear-sky RRTMG-LW using regridded ERA5 (Cartesian grid).

# Usage:
#     python RRTMG_example.py --utc 2025-11-01T06:00

import argparse
from pathlib import Path
import numpy as np
import xarray as xr
from climlab_rrtmg import rrtmg_lw


# =========================
# Default input NetCDF
# =========================
DEFAULT_NC = (
    "/home/xinyuewa/Research/Weather_model/UM-EARTH/"
    "33.04N_34.48N_107.48W_105.74W/"
    "regridded_ws-site1_20251102.nc"
)


# =========================
# Utilities
# =========================
def q_to_h2ovmr(q):
    #Specific humidity (kg/kg) -> H2O volume mixing ratio (mol/mol
    eps = 0.622
    q = np.asarray(q, dtype=np.float64)
    r = q / np.maximum(1.0 - q, 1e-12)
    return r / eps


def ensure_bottom_to_top(play, plev, tlay, q, x1, x1f):
    # Ensure pressure decreases bottom -> top
    if play[0] < play[-1]:
        play = play[::-1]
        tlay = tlay[::-1]
        q = q[::-1]
        x1 = x1[::-1]
        plev = plev[::-1]
        x1f = x1f[::-1]
    return play, plev, tlay, q, x1, x1f


# RRTMG LW wrapper
def run_rrtmg_lw(play, plev, tlay, tlev, tsfc, h2ovmr, co2_ppm=420.0):

    cp = 1004.0
    nbndlw = 16
    ngptlw = 140

    ncol, nlay = play.shape

    rrtmg_lw.climlab_rrtmg_lw_ini(cp)

    # gases
    o3vmr  = 1e-8 * np.ones_like(play)
    co2vmr = co2_ppm * 1e-6 * np.ones_like(play)
    ch4vmr = 1.8e-6 * np.ones_like(play)
    n2ovmr = 0.33e-6 * np.ones_like(play)
    o2vmr  = 0.2095 * np.ones_like(play)

    cfc11vmr = np.zeros_like(play)
    cfc12vmr = np.zeros_like(play)
    cfc22vmr = np.zeros_like(play)
    ccl4vmr  = np.zeros_like(play)

    # flags
    icld = 0
    ispec = 0
    idrv = 0
    inflglw, iceflglw, liqflglw = 2, 1, 1

    emis = np.ones((ncol, nbndlw))

    cldfmcl = np.zeros((ngptlw, ncol, nlay))
    taucmcl = np.zeros((ngptlw, ncol, nlay))
    ciwpmcl = np.zeros((ngptlw, ncol, nlay))
    clwpmcl = np.zeros((ngptlw, ncol, nlay))
    reicmcl = np.zeros((ncol, nlay))
    relqmcl = np.zeros((ncol, nlay))
    tauaer  = np.zeros((ncol, nlay, nbndlw))

    (olr_sr, uflx, dflx, hr,
     uflxc, dflxc, hrc, duflx_dt, duflxc_dt) = rrtmg_lw.climlab_rrtmg_lw(
        ncol, nlay, icld, ispec, idrv,
        play, plev, tlay, tlev, tsfc,
        h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr,
        cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr, emis,
        inflglw, iceflglw, liqflglw, cldfmcl,
        taucmcl, ciwpmcl, clwpmcl, reicmcl, relqmcl,
        tauaer
    )

    return {
        "olr_sr": np.array(olr_sr),
        "uflx": np.array(uflx),
        "dflx": np.array(dflx),
        "hr": np.array(hr),
        "cooling": -np.array(hr),
    }


# Main
def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--utc", required=True, help="UTC time, e.g. 2025-11-01T06:00")
    ap.add_argument("--nc", default=DEFAULT_NC)
    ap.add_argument("--co2-ppm", type=float, default=420.0)
    args = ap.parse_args()

    nc_path = Path(args.nc)
    ds = xr.open_dataset(nc_path)

    # select time
    utc = np.datetime64(args.utc)
    d0 = ds.sel(time=utc, method="nearest")

    # choose center column
    ix2 = ds.sizes["x2"] // 2
    ix3 = ds.sizes["x3"] // 2

    play_1d = d0["p"].isel(x2=ix2, x3=ix3).values
    plev_1d = d0["pressure_level"].isel(x2=ix2, x3=ix3).values
    tlay_1d = d0["t"].isel(x2=ix2, x3=ix3).values
    q_1d    = d0["q"].isel(x2=ix2, x3=ix3).values

    x1  = d0["x1"].values
    x1f = d0["x1f"].values

    play_1d, plev_1d, tlay_1d, q_1d, x1, x1f = ensure_bottom_to_top(
        play_1d, plev_1d, tlay_1d, q_1d, x1, x1f
    )

    tlev_1d = np.interp(x1f, x1, tlay_1d)
    tsfc = float(tlay_1d[0])
    h2ovmr_1d = q_to_h2ovmr(q_1d)

    play = play_1d[None, :]
    plev = plev_1d[None, :]
    tlay = tlay_1d[None, :]
    tlev = tlev_1d[None, :]
    h2ovmr = h2ovmr_1d[None, :]

    out = run_rrtmg_lw(
        play, plev, tlay, tlev, tsfc, h2ovmr,
        co2_ppm=args.co2_ppm
    )

    #  PRINT 
    print("=== RRTMG-LW clear-sky (single column) ===")
    print("file:", nc_path)
    print("requested UTC:", args.utc)
    print("selected time:", str(d0["time"].values))
    print("column:", f"x2={ix2}, x3={ix3}")
    print("nlay:", play.shape[1])
    print("play range (Pa):", float(play.min()), float(play.max()))

    #  Final OLR & cooling diagnostics 
    uflx = np.array(out["uflx"]).squeeze()
    dflx = np.array(out["dflx"]).squeeze()
    cool = np.array(out["cooling"])

    # TOA = minimum pressure level
    toa_idx = int(np.argmin(plev_1d))

    OLR = float(uflx[toa_idx])
    LWnet_toa = float(uflx[toa_idx] - dflx[toa_idx])
    
    hr_prof = np.array(out["hr"]).squeeze()
    hr_mean = float(np.mean(hr_prof))

    dp = np.abs(np.gradient(play_1d))
    hr_mean_pw = float(np.sum(hr_prof * dp) / np.sum(dp))

    print("mean HR (K/day):", hr_mean)
    print("pressure-weighted mean HR (K/day):", hr_mean_pw)


    print("OLR (TOA upwelling LW, W/m2):", OLR)
    print("TOA net LW (W/m2):", LWnet_toa)
    print("cooling min/max (K/day):",
        float(cool.min()), float(cool.max()))
    print("tsfc (K):", float(tsfc))
    print("tlay min/max (K):", float(tlay.min()), float(tlay.max()))
    print("h2ovmr min/max:", float(h2ovmr.min()), float(h2ovmr.max()))


    import csv

    utc_list = [
        "2025-11-01T00:00",
        "2025-11-01T06:00",
        "2025-11-01T12:00",
        "2025-11-01T18:00",
    ]

    rows = []
    for utc_str in utc_list:
        utc = np.datetime64(utc_str)
        d0 = ds.sel(time=utc, method="nearest")

        ix2 = ds.sizes["x2"] // 2
        ix3 = ds.sizes["x3"] // 2

        play_1d = d0["p"].isel(x2=ix2, x3=ix3).values
        plev_1d = d0["pressure_level"].isel(x2=ix2, x3=ix3).values
        tlay_1d = d0["t"].isel(x2=ix2, x3=ix3).values
        q_1d    = d0["q"].isel(x2=ix2, x3=ix3).values
        x1  = d0["x1"].values
        x1f = d0["x1f"].values

        play_1d, plev_1d, tlay_1d, q_1d, x1, x1f = ensure_bottom_to_top(
            play_1d, plev_1d, tlay_1d, q_1d, x1, x1f
        )

        tlev_1d = np.interp(x1f, x1, tlay_1d)
        tsfc = float(tlay_1d[0])
        h2ovmr_1d = q_to_h2ovmr(q_1d)

        play = play_1d[None, :]
        plev = plev_1d[None, :]
        tlay = tlay_1d[None, :]
        tlev = tlev_1d[None, :]
        h2ovmr = h2ovmr_1d[None, :]

        out = run_rrtmg_lw(play, plev, tlay, tlev, tsfc, h2ovmr, co2_ppm=420.0)

        uflx = np.array(out["uflx"]).squeeze()
        dflx = np.array(out["dflx"]).squeeze()
        cool = np.array(out["cooling"])

        toa_idx = int(np.argmin(plev_1d))
        OLR = float(uflx[toa_idx])
        LWnet_toa = float(uflx[toa_idx] - dflx[toa_idx])

        row = dict(
            requested_utc=utc_str,
            selected_time=str(d0["time"].values),
            tsfc_K=tsfc,
            tmin_K=float(tlay_1d.min()),
            tmax_K=float(tlay_1d.max()),
            h2ovmr_min=float(h2ovmr_1d.min()),
            h2ovmr_max=float(h2ovmr_1d.max()),
            OLR_Wm2=OLR,
            LWnet_TOA_Wm2=LWnet_toa,
            cooling_min_Kday=float(cool.min()),
            cooling_max_Kday=float(cool.max()),
        )
        rows.append(row)

        print(utc_str, "OLR:", OLR, "tsfc:", tsfc, "cool(max):", float(cool.max()))

    # save to CSV next to nc
    csv_path = nc_path.parent / "rrtmg_lw" / "rrtmg_lw_timeseries_ws-site1_20251101.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("saved:", csv_path)




if __name__ == "__main__":
    main()
