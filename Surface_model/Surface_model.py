#!/usr/bin/env python3
import argparse
import torch
from pathlib import Path
from datetime import datetime
import csv


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_LOCATIONS = PROJECT_ROOT / "locations.csv"
OUT_EVERY_SEC = 3600.0   # fixed hourly diagnostics


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



def parse_args():
    ap = argparse.ArgumentParser(description="Minimal surface model (step 1).")
    ap.add_argument("location_id", help="e.g. ws-site1")
    ap.add_argument("--nx", type=int, default=60, help="grid size in x")
    ap.add_argument("--ny", type=int, default=60, help="grid size in y")

    ap.add_argument("--dt", type=float, default=60.0, help="time step [s]")
    ap.add_argument("--t_end", type=float, default=86400.0, help="integration length [s]")

    ap.add_argument("--Ts0", type=float, default=250.0, help="initial surface temperature [K]")
    ap.add_argument("--albedo", type=float, default=0.1, help="surface albedo [-]")
    ap.add_argument("--Cp", type=float, default=1.0e7, help="surface heat capacity [J m^-2 K^-1]")

    ap.add_argument("--S0", type=float, default=1361.0, help="solar constant [W/m^2]")

    ap.add_argument("--utc", type=str, required=True,
                    help='UTC start time, e.g. "2025-11-01T06:00"')
    ap.add_argument("--locations_csv", type=Path, default=DEFAULT_LOCATIONS,
                    help="locations.csv with bbox (Latmin/Latmax/Lonmin/Lonmax)")
    ap.add_argument(
        "--save_every",
        type=float,
        default=3600.0,
        help="output snapshot interval [s], e.g. 3600 = hourly"
    )


    return ap.parse_args()


def insolation_sza(t_seconds, lat, device, dtype, S0=1361.0, day=86400.0):
    #  insolation using cos(SZA) at equinox (declination delta=0).
    #     H = 2π t / day
    #     mu = sin(lat)*sin(delta) + cos(lat)*cos(delta)*cos(H)
    #     clamp(mu, 0, +inf) so night is 0
    #  insolation using cos(SZA) at equinox (delta=0).
    # hour angle (scalar tensor)
    tsec = t_seconds if torch.is_tensor(t_seconds) else torch.tensor(t_seconds, device=device, dtype=dtype)
    H = 2.0 * torch.pi * (tsec / day - 0.5)

    delta = torch.tensor(0.0, device=device, dtype=dtype)  # equinox

    mu = torch.sin(lat) * torch.sin(delta) + torch.cos(lat) * torch.cos(delta) * torch.cos(H)
    mu = torch.clamp(mu, min=0.0)
    return S0 * mu  # same shape as lat: [Ny, Nx]


# Stefan–Boltzmann constant for surface thermal emission.
SIGMA = 5.670374419e-8  # W m^-2 K^-4

# surface energy balance
def surface_rhs(Ts, insol, albedo, Cp):
    # Absorbed solar radiation
    absorbed = (1.0 - albedo) * insol
    # Emitted thermal radiation
    # Right now Ts in the model is the surface effective temperature, because we assume a single-layer blackbody surface with no atmosphere.
    # Later, when we add atmospheric radiative transfer, Ts will still be the surface effective temperature, but it will differ from the planetary effective temperature
    emitted = SIGMA * Ts**4
    # Energy balance → temperature tendency:
    return (absorbed - emitted) / Cp # returning dT/dt.
# This is a third-order explicit Runge–Kutta (ATI3) time integrator used to advance surface temperature.
def rk3_step(T, dt, rhs, *rhs_args):
    k1 = rhs(T, *rhs_args)
    T1 = T + dt * k1

    k2 = rhs(T1, *rhs_args)
    T2 = 0.75 * T + 0.25 * (T1 + dt * k2)

    k3 = rhs(T2, *rhs_args)
    T3 = (1.0/3.0) * T + (2.0/3.0) * (T2 + dt * k3)
    return T3

def main():
    args = parse_args()
    outdir = Path(args.location_id)
    outdir.mkdir(parents=True, exist_ok=True)
    # Currently running on CPU; switching to GPU requires only one line change.
    device = torch.device("cpu")
    dtype = torch.float32

    ny, nx = args.ny, args.nx
    shape = (ny, nx)
    # ---- Read location center from locations.csv (bbox center) ----
    latc_deg, lonc_deg, bbox = load_location_center(args.locations_csv, args.location_id)
    print(f"[location] {args.location_id}: lat={latc_deg:.4f}, lon={lonc_deg:.4f}, bbox={bbox}")
    latmin, latmax, lonmin, lonmax = bbox

    # 1D -> 2D lat/lon grids (degrees)
    lat1d = torch.linspace(latmin, latmax, steps=ny, device=device, dtype=dtype)
    lon1d = torch.linspace(lonmin, lonmax, steps=nx, device=device, dtype=dtype)
    lat2d, lon2d = torch.meshgrid(lat1d, lon1d, indexing="ij")  # [ny,nx]

    # radians for trig
    lat_rad = lat2d * torch.pi / 180.0

    dt = args.dt
    t_end = args.t_end
    nsteps = int(t_end // dt)

    out_every = max(1, int(OUT_EVERY_SEC // dt))


    Ts = torch.full(shape, args.Ts0, device=device, dtype=dtype)
    albedo = torch.full(shape, args.albedo, device=device, dtype=dtype)
    Cp = torch.full(shape, args.Cp, device=device, dtype=dtype)

    lat = lat_rad


    # time series (hourly samples)
    t_hours_hist = []
    Ts_mean_hist = []
    insol_mean_hist = []
    dTsdt_mean_hist = []

    insol_min_day = torch.tensor(float("inf"), device=device, dtype=dtype)
    insol_max_day = torch.tensor(float("-inf"), device=device, dtype=dtype)
    dTsdt_last = None

    # 2D light-map stats over the whole day
    insol_max_map = torch.full(shape, -1.0, device=device, dtype=dtype)
    insol_sum_map = torch.zeros(shape, device=device, dtype=dtype)
    count_map = 0

   # ---- UTC -> local solar time offset (toy, ignores equation-of-time) ----
    dt_utc = datetime.fromisoformat(args.utc)
    utc_seconds = dt_utc.hour * 3600 + dt_utc.minute * 60 + dt_utc.second  # scalar

    # local solar seconds for each grid point (east positive lon)
    t_local0 = (utc_seconds + (lon2d / 15.0) * 3600.0) % 86400.0  # [ny,nx]

    print(f"[time] UTC={args.utc} | utc_seconds={utc_seconds} | lon_center={lonc_deg:.2f}")

    # At each time step:
    save_every = max(1, int(args.save_every // dt))  # e.g., hourly
    Ts_snap, insol_snap, dTsdt_snap, time_snap = [], [], [], []

    for n in range(nsteps):
        tau = n * dt  # lead time [s]
        tsec = (t_local0 + tau) % 86400.0  # [ny,nx] local solar seconds

        insol = insolation_sza(tsec, lat, device, dtype, S0=args.S0)  # [ny,nx]
        Ts = rk3_step(Ts, dt, surface_rhs, insol, albedo, Cp)
        # update min/max over the whole integration window
        insol_min_day = torch.minimum(insol_min_day, insol.min())
        insol_max_day = torch.maximum(insol_max_day, insol.max())
        insol_max_map = torch.maximum(insol_max_map, insol)
        insol_sum_map += insol
        count_map += 1

        # hourly snapshot
        if n % save_every == 0:
            dTsdt = surface_rhs(Ts, insol, albedo, Cp)  # [ny,nx]

            # --- preview like before (one line per hour) ---
            t_hr = tau / 3600.0
            insol_mean = float(insol.mean().item())
            Ts_mean = float(Ts.mean().item())
            dTsdt_mean = float(dTsdt.mean().item())

            print(
                f"t = {t_hr:5.1f} h | "
                f"insol = {insol_mean:8.2f} | "
                f"Ts_mean = {Ts_mean:8.3f} | "
                f"dTsdt_mean = {dTsdt_mean: .3e}"
            )
            # --- save snapshots (CPU) ---
            Ts_snap.append(Ts.detach().cpu())
            insol_snap.append(insol.detach().cpu())
            dTsdt_snap.append(dTsdt.detach().cpu())
            time_snap.append(float(tau))



    # --- stack to [Nt, ny, nx] ---
    Ts_3d = torch.stack(Ts_snap, dim=0)           # [Nt,ny,nx]
    insol_3d = torch.stack(insol_snap, dim=0)     # [Nt,ny,nx]
    dTsdt_3d = torch.stack(dTsdt_snap, dim=0)     # [Nt,ny,nx]
    lead_seconds = torch.tensor(time_snap, dtype=torch.float32)  # [Nt]

    # --- summary stats ---
    last_dTsdt_mean = float(dTsdt_3d[-1].mean().item()) if dTsdt_3d.numel() else float("nan")
    insol_min = float(insol_3d.min().item()) if insol_3d.numel() else float("nan")
    insol_max = float(insol_3d.max().item()) if insol_3d.numel() else float("nan")
    Ts_final_mean = float(Ts_3d[-1].mean().item()) if Ts_3d.numel() else float("nan")
    albedo_mean = float(albedo.mean().item())
    Cp_mean = float(Cp.mean().item())

    print("=== Surface model summary ===")
    print(f"dTsdt mean [K/s] (last reported): {last_dTsdt_mean: .6e}")
    print(f"insol min/max over integration: {insol_min:.3f} / {insol_max:.3f} W m^-2")
    print(f"Ts shape: ({ny}, {nx}) | Ts mean (final): {Ts_final_mean:.3f} K")
    print(f"albedo mean: {albedo_mean:.6f}")
    print(f"Cp mean: {Cp_mean:.6e} J m^-2 K^-1")
    print("=== End of summary ===")

    # --- buffer-style output (same design as topo) ---
    out = {
        "coords": {
            "start_utc": args.utc,          # string
            "lead_seconds": lead_seconds,   # [Nt]
            "lat_deg": lat2d.cpu(),         # [ny,nx]
            "lon_deg": lon2d.cpu(),         # [ny,nx]
        },
        "buffers": {
            "Ts_K": Ts_3d,                  # [Nt,ny,nx]
            "insol_Wm2": insol_3d,          # [Nt,ny,nx]
            "dTsdt_Ks": dTsdt_3d,           # [Nt,ny,nx]
            # (optional) keep local solar seconds if you want debug:
            # "t_local0_sec": t_local0.detach().cpu(),   # [ny,nx]
        },
        "params": {
            "dt": float(args.dt),
            "t_end": float(args.t_end),
            "save_every": float(args.save_every),
            "S0": float(args.S0),
            "albedo": float(args.albedo),
            "Cp": float(args.Cp),
            "bbox": bbox,  # (latmin,latmax,lonmin,lonmax)
        },
    }

    out_pt = outdir / f"{args.location_id}_surface_buffers.pt"
    torch.save(out, out_pt)
    print("Saved:", out_pt.resolve())




if __name__ == "__main__":
    main()
