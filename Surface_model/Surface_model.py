#!/usr/bin/env python3
import argparse
import torch
from pathlib import Path
from datetime import datetime, timedelta
import csv
import math


# File paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_LOCATIONS = PROJECT_ROOT / "locations.csv"

# Load location bbox from CSV
def load_location_center(csv_path, location_id):
    # Read bbox from locations.csv and return:
    #   (lat_center_deg, lon_center_deg, (latmin, latmax, lonmin, lonmax))

    # Supports:
    #   - comment lines starting with '#'
    #   - UTF-8 BOM
    #   - extra spaces around fields
    # utf-8-sig removes BOM if present
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        # filter out comment/blank lines BEFORE DictReader sees them
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]

    reader = csv.DictReader(lines)

    # normalize headers (strip spaces)
    if reader.fieldnames is None:
        raise ValueError(f"No header found in {csv_path}")
    reader.fieldnames = [h.strip() for h in reader.fieldnames]

    for row in reader:
        # normalize keys/values (strip spaces)
        row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}

        if row["Name"] == location_id:
            latmin = float(row["Latmin"])
            latmax = float(row["Latmax"])
            lonmin = float(row["Lonmin"])
            lonmax = float(row["Lonmax"])

            latc = 0.5 * (latmin + latmax)
            lonc = 0.5 * (lonmin + lonmax)
            return latc, lonc, (latmin, latmax, lonmin, lonmax)

    raise ValueError(f"Location '{location_id}' not found in {csv_path}")

def solar_declination_rad_from_utc(utc_str: str) -> float:
    # Very simple solar declination model (no equation-of-time).
    # Input: ISO UTC string like "2025-11-01T06:00"
    # Output: declination in radians
    dt_utc = datetime.fromisoformat(utc_str)
    doy = dt_utc.timetuple().tm_yday  # 1..365/366

    # Common approximation:
    # delta = 23.44° * sin( 2π * (doy - 81) / 365 )
    delta_deg = 23.44 * math.sin(2.0 * math.pi * (doy - 81) / 365.0)
    return math.radians(delta_deg)

def parse_args():
    ap = argparse.ArgumentParser(description="Minimal surface model (step 1).")
    ap.add_argument("location_id", help="e.g. ws-site1")
    ap.add_argument("--nx", type=int, default=60, help="grid size in x")
    ap.add_argument("--ny", type=int, default=60, help="grid size in y")

    ap.add_argument("--dt", type=float, default=60.0, help="time step [s]")
    ap.add_argument("--t_end", type=float, default=3 * 24 * 3600, help="integration length [s]")

    ap.add_argument("--Ts0", type=float, default=250.0, help="initial surface temperature [K]")
    ap.add_argument("--albedo", type=float, default=0.1, help="surface albedo [-]")
    # --- two-layer surface (skin + subsurface) ---
    ap.add_argument("--C1", type=float, default=2.0e6,
                    help="skin heat capacity [J m^-2 K^-1] (fast layer)")
    ap.add_argument("--C2", type=float, default=2.0e7,
                    help="subsurface heat capacity [J m^-2 K^-1] (slow layer)")
    ap.add_argument("--k", type=float, default=15.0,
                    help="coupling/thermal conductance [W m^-2 K^-1] between layers")

    ap.add_argument("--S0", type=float, default=1361.0, help="solar constant [W/m^2]")

    ap.add_argument("--utc", type=str, required=True,
                    help='UTC start time, default use 00Z for ERA alignment, e.g. "2025-11-01T00:00"')
    ap.add_argument("--locations_csv", type=Path, default=DEFAULT_LOCATIONS,
                    help="locations.csv with bbox (Latmin/Latmax/Lonmin/Lonmax)")
    ap.add_argument(
        "--save_every", 
        type=float,
        default=21600.0,
        help="output snapshot interval [s], default 21600 (= 6-hourly, matches ERA 00/06/12/18)"
    )


    return ap.parse_args()


def insolation_sza(t_seconds, lat, delta_rad, device, dtype, S0=1361.0, day=86400.0):
    #  insolation using cos(SZA) 
    #     H = 2π t / day
    #     mu = sin(lat)*sin(delta) + cos(lat)*cos(delta)*cos(H)
    #     clamp(mu, 0, +inf) so night is 0
    #  insolation using cos(SZA) 
    # hour angle (scalar tensor)
    tsec = t_seconds if torch.is_tensor(t_seconds) else torch.tensor(t_seconds, device=device, dtype=dtype)
    H = 2.0 * torch.pi * (tsec / day - 0.5)

    delta = torch.tensor(delta_rad, device=device, dtype=dtype) # solar declination in radians (scalar)

    mu = torch.sin(lat) * torch.sin(delta) + torch.cos(lat) * torch.cos(delta) * torch.cos(H)
    mu = torch.clamp(mu, min=0.0)
    return S0 * mu  # same shape as lat: [Ny, Nx]


# Stefan–Boltzmann constant for surface thermal emission.
SIGMA = 5.670374419e-8  # W m^-2 K^-4

# surface energy balance
def surface_rhs_2layer(T1, T2, insol, albedo, C1, C2, k):
    # Two-layer surface energy balance:
    #   C1 dT1/dt = (1-a)S - sigma*T1^4 - k(T1-T2)
    #   C2 dT2/dt =  k(T1-T2)
    absorbed = (1.0 - albedo) * insol
    emitted = SIGMA * T1**4
    flux_cond = k * (T1 - T2)

    dT1dt = (absorbed - emitted - flux_cond) / C1
    dT2dt = (flux_cond) / C2
    return dT1dt, dT2dt

# This is a third-order explicit Runge–Kutta (ATI3) time integrator used to advance surface temperature.
def rk3_step_2(T1, T2, dt, rhs2, *rhs_args):
    k1_1, k1_2 = rhs2(T1, T2, *rhs_args)
    T1a = T1 + dt * k1_1
    T2a = T2 + dt * k1_2

    k2_1, k2_2 = rhs2(T1a, T2a, *rhs_args)
    T1b = 0.75 * T1 + 0.25 * (T1a + dt * k2_1)
    T2b = 0.75 * T2 + 0.25 * (T2a + dt * k2_2)

    k3_1, k3_2 = rhs2(T1b, T2b, *rhs_args)
    T1n = (1.0/3.0) * T1 + (2.0/3.0) * (T1b + dt * k3_1)
    T2n = (1.0/3.0) * T2 + (2.0/3.0) * (T2b + dt * k3_2)
    return T1n, T2n

def main():
    args = parse_args()
    outdir = Path(args.location_id)
    outdir.mkdir(parents=True, exist_ok=True)
    # Currently running on CPU; switching to GPU requires only one line change.
    device = torch.device("cpu")
    dtype = torch.float32

    ny, nx = args.ny, args.nx
    shape = (ny, nx)
    # Read location center from locations.csv (bbox center) 
    latc_deg, lonc_deg, bbox = load_location_center(args.locations_csv, args.location_id)
    print(f"[location] {args.location_id}: lat={latc_deg:.4f}, lon={lonc_deg:.4f}, bbox={bbox}")
    latmin, latmax, lonmin, lonmax = bbox

    # 2D lat/lon grids (degrees)
    lat1d = torch.linspace(latmin, latmax, steps=ny, device=device, dtype=dtype)
    lon1d = torch.linspace(lonmin, lonmax, steps=nx, device=device, dtype=dtype)
    lat2d, lon2d = torch.meshgrid(lat1d, lon1d, indexing="ij")  # [ny,nx]

    # radians for trig
    lat_rad = lat2d * torch.pi / 180.0

    dt = args.dt
    t_end = args.t_end
    nsteps = int(t_end // dt) + 1   # include endpoint




    T1 = torch.full(shape, args.Ts0, device=device, dtype=dtype)  # skin
    T2 = torch.full(shape, args.Ts0, device=device, dtype=dtype)  # subsurface (start same)

    albedo = torch.full(shape, args.albedo, device=device, dtype=dtype)
    C1 = torch.full(shape, args.C1, device=device, dtype=dtype)
    C2 = torch.full(shape, args.C2, device=device, dtype=dtype)
    k  = torch.full(shape, args.k,  device=device, dtype=dtype)


    lat = lat_rad


    # time series 
    insol_snap, time_snap = [],[]
    T1_snap, T2_snap = [], []
    dT1dt_snap, dT2dt_snap = [], []
    Qsw_snap, Qlw_snap, Qg_snap, Qnet_snap = [], [], [], []
    insol_min_day = torch.tensor(float("inf"), device=device, dtype=dtype)
    insol_max_day = torch.tensor(float("-inf"), device=device, dtype=dtype)

   # UTC -> local solar time offset (toy, ignores equation-of-time) 
    dt_utc = datetime.fromisoformat(args.utc)
    delta_rad = solar_declination_rad_from_utc(args.utc)
    print(f"[sun] declination delta = {delta_rad * 180.0 / math.pi:.3f} deg")
    utc_seconds = dt_utc.hour * 3600 + dt_utc.minute * 60 + dt_utc.second  # scalar

    # local solar seconds for each grid point (east positive lon)
    t_local0 = (utc_seconds + (lon2d / 15.0) * 3600.0) % 86400.0  # [ny,nx]

    print(f"[time] UTC={args.utc} | utc_seconds={utc_seconds} | lon_center={lonc_deg:.2f}")

    # At each time step:
   
    for n in range(nsteps):
        tau = int(n * dt)  # lead time [s]
        tsec = (t_local0 + tau) % 86400.0  # [ny,nx] local solar seconds

        insol = insolation_sza(tsec, lat, delta_rad, device, dtype, S0=args.S0)  # [ny,nx]

        # --- advance Ts ---
        T1, T2 = rk3_step_2(T1, T2, dt, surface_rhs_2layer, insol, albedo, C1, C2, k)


        # update min/max over the whole integration window (use insol at this step)
        insol_min_day = torch.minimum(insol_min_day, insol.min())
        insol_max_day = torch.maximum(insol_max_day, insol.max())

        # snapshot every save_every seconds
        if (tau % args.save_every) == 0:
            # use dTsdt_old (aligned with insol at this step)
            dT1dt, dT2dt = surface_rhs_2layer(T1, T2, insol, albedo, C1, C2, k)
            # --- flux diagnostics (W/m^2) ---
            Q_sw_abs = (1.0 - albedo) * insol          # absorbed shortwave
            Q_lw_up  = SIGMA * T1**4                   # upward longwave emission (surface)
            Q_g      = k * (T1 - T2)                   # ground heat flux (downward positive if T1>T2)
            Q_net    = Q_sw_abs - Q_lw_up - Q_g        # net energy into surface layer

            t_hr = tau / 3600.0
            insol_mean = float(insol.mean().item())
            Ts_mean = float(T1.mean().item())      # surface temp = skin
            T2_mean = float(T2.mean().item())
            dT1dt_mean = float(dT1dt.mean().item())
            Qnet_mean = float(Q_net.mean().item())
            Qg_mean   = float(Q_g.mean().item())


            print(
                f"t = {t_hr:5.1f} h | "
                f"insol = {insol_mean:8.2f} | "
                f"T1_mean = {Ts_mean:8.3f} | "
                f"T2_mean = {T2_mean:8.3f} | "
                f"dT1dt_mean = {dT1dt_mean: .3e} | "
                f"Qnet = {Qnet_mean:7.1f} | "
                f"Qg = {Qg_mean:7.1f}"
            )

            T1_snap.append(T1.detach().cpu())
            T2_snap.append(T2.detach().cpu())
            insol_snap.append(insol.detach().cpu())
            dT1dt_snap.append(dT1dt.detach().cpu())
            dT2dt_snap.append(dT2dt.detach().cpu())
            time_snap.append(float(tau))
            Qsw_snap.append(Q_sw_abs.detach().cpu())
            Qlw_snap.append(Q_lw_up.detach().cpu())
            Qg_snap.append(Q_g.detach().cpu())
            Qnet_snap.append(Q_net.detach().cpu())





    #  stack to [Nt, ny, nx] 
    T1_3d = torch.stack(T1_snap, dim=0)
    T2_3d = torch.stack(T2_snap, dim=0)          # [Nt,ny,nx]
    insol_3d = torch.stack(insol_snap, dim=0)     # [Nt,ny,nx]
    dT1dt_3d = torch.stack(dT1dt_snap, dim=0)     # [Nt,ny,nx]
    dT2dt_3d = torch.stack(dT2dt_snap, dim=0)     # [Nt,ny,nx]
    lead_seconds = torch.tensor(time_snap, dtype=torch.float32)  # [Nt]
    Qsw_3d  = torch.stack(Qsw_snap, dim=0)
    Qlw_3d  = torch.stack(Qlw_snap, dim=0)
    Qg_3d   = torch.stack(Qg_snap, dim=0)
    Qnet_3d = torch.stack(Qnet_snap, dim=0)
    # --- surface forcing products (for atmospheric model) ---
    # surface -> atmosphere net heating (no H/LE yet)
    Qatm_3d = Qsw_3d - Qlw_3d - Qg_3d

    # surface (skin) temperature for Dirichlet BC
    Tsfc_3d = T1_3d

    #  summary stats 
    last_dTsdt_mean = float(dT1dt_3d[-1].mean().item()) if dT1dt_3d.numel() else float("nan")
    insol_min = float(insol_3d.min().item()) if insol_3d.numel() else float("nan")
    insol_max = float(insol_3d.max().item()) if insol_3d.numel() else float("nan")
    T1_final_mean = float(T1_3d[-1].mean().item()) if T1_3d.numel() else float("nan")
    T2_final_mean = float(T2_3d[-1].mean().item()) if T2_3d.numel() else float("nan")
    albedo_mean = float(albedo.mean().item())
    C1_mean = float(C1.mean().item())
    C2_mean = float(C2.mean().item())
    k_mean = float(k.mean().item())
    print("=== Surface model summary ===")
    print(f"dT1dt mean [K/s] (last reported): {last_dTsdt_mean: .6e}")
    print(f"insol min/max over integration: {insol_min:.3f} / {insol_max:.3f} W m^-2")
    print(f"T1 shape: ({ny}, {nx}) | T1 mean (final): {T1_final_mean:.3f} K")
    print(f"T2 shape: ({ny}, {nx}) | T2 mean (final): {T2_final_mean:.3f} K")
    print(f"albedo mean: {albedo_mean:.6f}")
    print(f"C1 mean: {C1_mean:.6e} J m^-2 K^-1")
    print(f"C2 mean: {C2_mean:.6e} J m^-2 K^-1")
    print(f"k mean:  {k_mean:.6e} W m^-2 K^-1")
    print(f"Qnet mean (final): {float(Qnet_3d[-1].mean().item()):.2f} W m^-2")
    print(f"Qg   mean (final): {float(Qg_3d[-1].mean().item()):.2f} W m^-2")


    print("=== End of summary ===")
    # build UTC timestamps aligned with lead_seconds
    start_dt = datetime.fromisoformat(args.utc)
    valid_time_utc = [
        (start_dt + timedelta(seconds=float(s))).isoformat(timespec="seconds")
        for s in lead_seconds.tolist()
    ]
    #  buffer-style output (same design as topo) 
    out = {
        "coords": {
            "start_utc": args.utc,          # string
            "valid_time_utc": valid_time_utc,  # [Nt] list of strings
            "lead_seconds": lead_seconds,   # [Nt]
            "lat_deg": lat2d.cpu(),         # [ny,nx]
            "lon_deg": lon2d.cpu(),         # [ny,nx]
        },
        "buffers": {
            "T1_K": T1_3d,
            "T2_K": T2_3d,
            "insol_Wm2": insol_3d,
            "dT1dt_Ks": dT1dt_3d,
            "dT2dt_Ks": dT2dt_3d,
            "Qsw_abs_Wm2": Qsw_3d,
            "Qlw_up_Wm2":  Qlw_3d,
            "Qg_Wm2":      Qg_3d,
            "Qnet_Wm2":    Qnet_3d,
            "Qatm_Wm2":    Qatm_3d,
            "Tsfc_K":      Tsfc_3d,


            # (optional) keep local solar seconds if you want debug:
            # "t_local0_sec": t_local0.detach().cpu(),   # [ny,nx]
        },
        "params": {
            "dt": float(args.dt),
            "t_end": float(args.t_end),
            "save_every": float(args.save_every),
            "S0": float(args.S0),
            "albedo": float(args.albedo),
            "C1": float(args.C1),
            "C2": float(args.C2),
            "k": float(args.k),
            "bbox": bbox,
        },
    }

    out_pt = outdir / f"{args.location_id}_surface_buffers.pt"
    torch.save(out, out_pt)
    chk = torch.load(out_pt, map_location="cpu")
    print("[check] keys:", chk.keys())
    print("[check] buffers keys:", chk["buffers"].keys())
    print("[check] valid_time_utc:", chk["coords"]["valid_time_utc"])
    for k in ["T1_K", "T2_K", "Qnet_Wm2", "Qg_Wm2", "Qsw_abs_Wm2", "Qlw_up_Wm2"]:
        if k in chk["buffers"]:
            v = chk["buffers"][k]
            print(f"[check] {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"[check] MISSING buffer: {k}")

    print("Saved:", out_pt.resolve())
if __name__ == "__main__":
    main()
