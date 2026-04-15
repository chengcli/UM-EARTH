import argparse
import os
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import snapy
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d
import torch.nn.functional as F
import yaml
from kintera import Kinetics, KineticsOptions, ThermoX
from paddle import evolve_kinetics
from snapy import Mesh, MeshBlock, MeshBlockOptions, MeshOptions
from snapy import kICY, kIDN, kIPR, kIV1, kIV2, kIV3

from refine_utils import coarsen_spatial, conservative_refine, refine_spatial


torch.set_default_dtype(torch.float64)

TOPO_SEQUENCE = ("2p4km", "1p2km", "0p6km", "0p3km")


@dataclass
class ForecastContext:
    mesh: Mesh
    thermo_x: ThermoX
    kinet: Kinetics
    device: torch.device
    block_vars: dict[str, torch.Tensor]
    ecmwf_hydro_u: list[torch.Tensor]
    stage_topography: list[dict[str, torch.Tensor]]
    precip_indices: tuple[int, ...]
    current_time: float = 0.0
    refine_at_18h: bool = False


@dataclass(frozen=True)
class ForcingSchedule:
    interval_seconds: float
    next_index: int = 1


def implicit_scheme_for_refinement_factor(refinement_factor: float) -> int:
    return 1


def refined_global_horizontal_cells(
    current_local_nx2: int,
    current_local_nx3: int,
    px: int,
    py: int,
) -> tuple[int, int]:
    global_nx2 = max(current_local_nx2, 1) * max(px, 1)
    global_nx3 = max(current_local_nx3, 1) * max(py, 1)
    next_nx2 = global_nx2 * 2 if global_nx2 > 1 else 1
    next_nx3 = global_nx3 * 2 if global_nx3 > 1 else 1
    return next_nx2, next_nx3

def print_ok(*args):
    message = " ".join(str(arg) for arg in args)
    print("\033[92m[OK]\033[0m ", message, flush=True)

def print_err(*args):
    message = " ".join(str(arg) for arg in args)
    print("\033[91m[ERR]\033[0m ", message, flush=True)

def call_user_output(bvars: dict[str, torch.Tensor]):
    hydro_w = bvars["hydro_w"]
    out = {}
    out["qtol"] = hydro_w[kICY:].sum(dim=0)
    return out

def infer_run_dir(config_file: str) -> Path:
    return Path(config_file).resolve().parent


def resolve_restart_file(input_path: str) -> Path:
    candidate = Path(input_path).resolve()
    if candidate.is_file():
        if candidate.suffix not in {".part", ".restart"}:
            raise FileNotFoundError(f"Expected a .part or .restart input, found {candidate}")
        return candidate

    if not candidate.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {candidate}")

    preferred_restart = candidate / "input.restart"
    if preferred_restart.exists():
        return preferred_restart

    restart_matches = sorted(candidate.glob("*.restart"))
    if restart_matches:
        if len(restart_matches) > 1:
            print_ok("Multiple restart bundles found; using", restart_matches[0].name)
        return restart_matches[0]

    preferred = candidate / "input.part"
    if preferred.exists():
        return preferred

    matches = sorted(candidate.glob("*.part"))
    if not matches:
        raise FileNotFoundError(f"Could not find any .part restart tensor in {candidate}")
    if len(matches) > 1:
        print_ok("Multiple restart tensors found; using", matches[0].name)
    return matches[0]


def current_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def load_ranked_restart_module(restart_file: Path, rank: int):
    if restart_file.suffix == ".part":
        return torch.jit.load(str(restart_file))
    if restart_file.suffix != ".restart":
        raise ValueError(f"Unsupported restart input type: {restart_file}")

    with tarfile.open(restart_file, "r") as archive:
        members = sorted(
            member for member in archive.getnames()
            if member.endswith(f".block{rank}.00000.part")
            or (f".block{rank}." in member and member.endswith(".part"))
        )
        if not members:
            raise FileNotFoundError(
                f"Could not find block{rank} part file in restart bundle {restart_file}"
            )
        member = members[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            archive.extract(member, path=tmpdir, filter="data")
            part_path = Path(tmpdir) / member
            return torch.jit.load(str(part_path))


def default_output_dir(config_file: str) -> Path:
    return infer_run_dir(config_file) / "forecast_output"


def default_input_dir(config_file: str) -> Path:
    run_dir = infer_run_dir(config_file)
    matches = sorted(run_dir.glob("*/**/regridded_*_tensors"))
    if not matches:
        matches = sorted(run_dir.glob("forecast_input/regridded_*_tensors"))
    if not matches:
        raise FileNotFoundError(
            f"Could not infer prepared tensor directory under {run_dir}"
        )
    if len(matches) > 1:
        print_ok("Multiple tensor directories found; using", matches[0])
    return matches[0]


def parse_topography_label(path: Path) -> str:
    marker = "_topo_"
    if marker not in path.stem:
        raise ValueError(f"Topography tensor does not follow '*_topo_<label>.pt': {path}")
    return path.stem.split(marker, 1)[1]


def discover_topography_files(
    topography_dir: str,
    required_labels: tuple[str, ...] = TOPO_SEQUENCE,
) -> dict[str, Path]:
    topo_root = Path(topography_dir).resolve()
    topo_paths = sorted(topo_root.glob("*_topo_*.pt"))
    if not topo_paths:
        raise FileNotFoundError(f"Could not find any topography tensors in {topo_root}")

    discovered = {parse_topography_label(path): path for path in topo_paths}
    mapping = {label: discovered[label] for label in required_labels if label in discovered}
    missing = [label for label in required_labels if label not in mapping]
    if missing:
        raise FileNotFoundError(
            f"Missing topography tensors for {missing} in {topo_root}; found {sorted(mapping)}"
        )
    return mapping


def default_topography_dir(config_file: str) -> Path:
    return infer_run_dir(config_file) / "topography" / "products"


def precipitation_tracer_indices(config_file: str) -> tuple[int, ...]:
    with open(config_file, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    species = config.get("species", [])
    explicit_species = species[1:]
    indices = []
    for offset, spec in enumerate(explicit_species):
        name = str(spec.get("name", ""))
        if "(p)" in name or ",p)" in name:
            indices.append(kICY + offset)
    return tuple(indices)


def init_dist(backend: str, requested_device: str = "auto") -> torch.device:
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", os.environ["RANK"])

    local_rank = int(os.environ["LOCAL_RANK"])
    if requested_device == "cpu":
        backend = "gloo"

    if backend == "gloo":
        dist.init_process_group(backend="gloo", init_method="env://")
        device = torch.device("cpu")
    elif backend == "nccl":
        if requested_device == "cpu":
            dist.init_process_group(backend="gloo", init_method="env://")
            device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL backend requires CUDA")
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            dist.init_process_group(
                backend="cpu:gloo,cuda:nccl", device_id=device, init_method="env://"
            )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    snapy.distributed.set_process_group(dist_c10d._get_default_group())
    return device


def create_mesh(
    config_file: str,
    output_dir: str,
    requested_device: str = "auto",
):
    options = MeshOptions.from_yaml(config_file, verbose=False)
    options.block().output_dir(str(Path(output_dir).resolve()))
    options.block().basename(Path(config_file).stem)

    if requested_device == "cpu":
        options.block().layout().backend("gloo")

    backend = options.block().layout().backend()
    device = init_dist(backend, requested_device=requested_device)
    print_ok("Using device =", device, "with backend =", backend)

    mesh = Mesh(options)
    mesh_device = torch.device(mesh.options.device_str() or str(device))
    mesh.to(mesh_device)
    for block in mesh.blocks:
        block.set_user_output_func(call_user_output)

    block0 = mesh.blocks[0]
    thermo_y = block0.module("hydro.eos.thermo")
    thermo_x = ThermoX(thermo_y.options)
    thermo_x.to(mesh_device)

    # kinetics model
    op_kinet = KineticsOptions.from_yaml(config_file)
    kinet = Kinetics(op_kinet)
    kinet.to(mesh_device)

    return mesh, thermo_x, kinet, mesh_device

def load_restart_slice(
    restart_file: Path,
    index: int = 0,
    device: torch.device = torch.device("cpu"),
):
    module = load_ranked_restart_module(restart_file, current_rank())
    block_vars = {}
    for name, data in module.named_buffers(recurse=True):
        if index >= data.shape[0]:
            raise IndexError(f"Restart index {index} is out of range for {name} with shape {tuple(data.shape)}")
        block_vars[name] = data[index].to(device).to(torch.double)

    for name, data in module.named_parameters(recurse=True):
        if index >= data.shape[0]:
            raise IndexError(f"Restart index {index} is out of range for {name} with shape {tuple(data.shape)}")
        block_vars[name] = data[index].to(device).to(torch.double)
    return block_vars


def load_topography_module(
    topo_file: Path,
    device: torch.device = torch.device("cpu"),
):
    module = torch.jit.load(str(topo_file))
    aux_vars = {}
    for name, data in module.named_buffers(recurse=True):
        aux_vars[name] = data.to(device).to(torch.double)
    for name, data in module.named_parameters(recurse=True):
        aux_vars[name] = data.to(device).to(torch.double)
    return aux_vars


def orient_topography(topo_vars: dict[str, torch.Tensor]) -> torch.Tensor:
    topo = topo_vars["topography"]
    row0_is_north = topo_vars.get("row0_is_north")
    if row0_is_north is not None and bool(row0_is_north.item()):
        topo = torch.flip(topo, dims=(0,))
    return topo


def resample_topography(
    topo_2d: torch.Tensor,
    target_shape: tuple[int, int],
) -> torch.Tensor:
    if tuple(topo_2d.shape) == target_shape:
        return topo_2d

    sampled = F.interpolate(
        topo_2d.unsqueeze(0).unsqueeze(0),
        size=target_shape,
        mode="bilinear",
        align_corners=False,
    )
    return sampled.squeeze(0).squeeze(0)


def pad_topography(topo_2d: torch.Tensor, nghost: int) -> torch.Tensor:
    padded = F.pad(
        topo_2d.unsqueeze(0).unsqueeze(0),
        pad=(nghost, nghost, nghost, nghost),
        mode="replicate",
    )
    return padded.squeeze(0).squeeze(0).transpose(0, 1).unsqueeze(-1)


def build_topography_field(
    topo_vars: dict[str, torch.Tensor],
    interior_shape: tuple[int, int],
    nghost: int,
) -> torch.Tensor:
    topo_2d = orient_topography(topo_vars)
    topo_2d = resample_topography(topo_2d, interior_shape)
    return pad_topography(topo_2d, nghost)


def load_time_series_inputs(
    restart_file: Path,
    topography_files: dict[str, Path],
    device: torch.device,
):
    time_slices = [load_restart_slice(restart_file, index=n, device=device) for n in range(4)]
    topography = {
        label: load_topography_module(path, device=device)
        for label, path in topography_files.items()
    }
    return time_slices, topography


def selected_topography_labels(
    refinement_mode: str,
    topography_stage: str | None,
) -> tuple[str, ...]:
    if refinement_mode == "staged":
        return TOPO_SEQUENCE
    return (topography_stage or TOPO_SEQUENCE[0],)


def build_topography_field_for_block(
    block: MeshBlock,
    topo_vars: dict[str, torch.Tensor],
) -> torch.Tensor:
    coord = block.module("coord")
    topo_2d = orient_topography(topo_vars).to(coord.buffer("x2v").device).to(torch.double)
    x2v = coord.buffer("x2v").to(torch.double)
    x3v = coord.buffer("x3v").to(torch.double)
    layout = block.get_layout()
    rank = block.options.layout().rank()
    lx2, lx3, _ = layout.loc_of(rank)
    coord_opts = block.options.coord()
    width_x2 = coord_opts.x2max() - coord_opts.x2min()
    width_x3 = coord_opts.x3max() - coord_opts.x3min()
    global_x2min = coord_opts.x2min() - lx2 * width_x2
    global_x2max = global_x2min + width_x2 * block.options.layout().px()
    global_x3min = coord_opts.x3min() - lx3 * width_x3
    global_x3max = global_x3min + width_x3 * block.options.layout().py()

    x2_frac = ((x2v - global_x2min) / max(global_x2max - global_x2min, 1.0e-12)).clamp(0.0, 1.0)
    x3_frac = ((x3v - global_x3min) / max(global_x3max - global_x3min, 1.0e-12)).clamp(0.0, 1.0)
    grid_x2, grid_x3 = torch.meshgrid(x2_frac * 2.0 - 1.0, x3_frac * 2.0 - 1.0, indexing="ij")
    grid = torch.stack((grid_x3, grid_x2), dim=-1).unsqueeze(0)
    sampled = F.grid_sample(
        topo_2d.unsqueeze(0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(0).transpose(0, 1).unsqueeze(-1)

def equilibrate_initial_fields(block: MeshBlock,
                               thermo_x: ThermoX,
                               block_vars: dict[str, torch.Tensor]):
    thermo_y = block.module("hydro.eos.thermo")

    hydro_w = block_vars["hydro_w"]
    dens = hydro_w[kIDN]
    pres = hydro_w[kIPR]
    ivol = thermo_y.compute("DY->V", (dens, hydro_w[kICY:]))
    temp = thermo_y.compute("PV->T", (pres, ivol))
    xfrac = thermo_y.compute("Y->X", (hydro_w[kICY:],))

    thermo_x(temp, pres, xfrac)
    hydro_w[kICY:] = thermo_x.compute("X->Y", (xfrac,))

    return block_vars


def surface_precipitation_mask(topo: torch.Tensor) -> torch.Tensor:
    surface = torch.zeros_like(topo, dtype=torch.bool)
    if topo.shape[-1] == 0:
        return surface

    surface[..., 0] = ~topo[..., 0]
    surface[..., 1:] = (~topo[..., 1:]) & topo[..., :-1]
    return surface


def remove_surface_precipitation(
    hydro_w: torch.Tensor,
    topo: torch.Tensor,
    precip_indices: tuple[int, ...],
) -> torch.Tensor:
    if not precip_indices:
        return hydro_w

    surface = surface_precipitation_mask(topo)
    if not torch.any(surface):
        return hydro_w

    updated = hydro_w.clone()
    for index in precip_indices:
        updated[index] = torch.where(surface, torch.zeros_like(updated[index]), updated[index])
    return updated


def next_forcing_time(schedule: ForcingSchedule, available_slices: int) -> float | None:
    if schedule.next_index >= available_slices:
        return None
    return schedule.next_index * schedule.interval_seconds


def clone_with_lateral_ghost_zones(
    state: torch.Tensor,
    boundary_state: torch.Tensor,
    nghost: int,
) -> torch.Tensor:
    if nghost <= 0:
        return state
    if state.shape != boundary_state.shape:
        raise ValueError(
            f"State shape mismatch: {tuple(state.shape)} vs {tuple(boundary_state.shape)}"
        )

    updated = state.clone()
    updated[..., :nghost, :, :] = boundary_state[..., :nghost, :, :]
    updated[..., -nghost:, :, :] = boundary_state[..., -nghost:, :, :]
    updated[..., :, :nghost, :] = boundary_state[..., :, :nghost, :]
    updated[..., :, -nghost:, :] = boundary_state[..., :, -nghost:, :]
    return updated


def refine_boundary_state_to_match(
    boundary_state: torch.Tensor,
    target_shape: torch.Size | tuple[int, ...],
    nghost: int,
) -> torch.Tensor:
    target_shape = tuple(target_shape)
    if tuple(boundary_state.shape) == target_shape:
        return boundary_state

    refined = boundary_state
    while tuple(refined.shape) != target_shape:
        next_shape = list(refined.shape)
        for dim in (-3, -2):
            cells = next_shape[dim]
            if cells <= 1:
                continue
            next_shape[dim] = (cells - 2 * nghost) * 2 + 2 * nghost
        next_shape = tuple(next_shape)
        if any(dim <= 0 for dim in next_shape):
            raise ValueError(
                f"Cannot refine boundary state from {tuple(refined.shape)} with nghost={nghost}"
            )

        trailing_shape = refined.shape[-3:]
        sampled = F.interpolate(
            refined.reshape(-1, 1, *trailing_shape),
            size=next_shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        refined = sampled.reshape(*next_shape)
        if any(current > target for current, target in zip(refined.shape, target_shape)):
            raise ValueError(
                "Refined boundary state exceeded target shape: "
                f"{tuple(refined.shape)} vs {target_shape}"
            )

    return refined


def apply_ghost_zone_forcing(
    mesh: Mesh,
    block_vars: dict[str, torch.Tensor],
    target_hydro_u: torch.Tensor,
) -> dict[str, torch.Tensor]:
    nghost = mesh.blocks[0].options.coord().nghost()
    block_vars["hydro_u"] = clone_with_lateral_ghost_zones(
        block_vars["hydro_u"], target_hydro_u, nghost
    )
    return synchronize_hydro_state(mesh, block_vars)


def synchronize_hydro_state(
    mesh: Mesh,
    block_vars: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    eos = mesh.blocks[0].module("hydro.eos")
    hydro_w = eos.compute("U->W", (block_vars["hydro_u"],))
    mesh.blocks[0].apply_hydro_bc(hydro_w)
    block_vars["hydro_w"] = hydro_w
    return block_vars

def add_frigate_forcing(block_vars: dict[str, torch.Tensor],
                        eos,
                        precip_indices: tuple[int, ...],
                        dt: float, 
                        tau: float = 100.,
                        cooling_rate: float = 2. / 86400.):
    topo = block_vars["topo"]
    w = block_vars["hydro_w"]
    u = block_vars["hydro_u"]
    rho = u[kIDN]
    cv = 717.5

    # boundary layer forcing
    u[kIV1] -= rho * w[kIV1] * topo.int() * dt / tau
    u[kIV2] -= rho * w[kIV2] * topo.int() * dt / tau
    u[kIV3] -= rho * w[kIV3] * topo.int() * dt / tau

    # uniform cooling
    #u[IPR] -= rho * cv * cooling_rate * (1. - topo.int()) * dt

    # surface heating
    # surface_level = torch.zeros_like(topo)
    # surface_level[:,:,1:] = topo[:,:,:-1] - topo[:,:,1:]

    updated_hydro_w = remove_surface_precipitation(w, topo, precip_indices)
    if updated_hydro_w.data_ptr() != w.data_ptr():
        block_vars["hydro_w"] = updated_hydro_w
        block_vars["hydro_u"] = eos.compute("W->U", (updated_hydro_w,))

def run_simulation(mesh: Mesh,
                   thermo_x: ThermoX,
                   kinet: Kinetics,
                   mesh_vars: list[dict[str, torch.Tensor]],
                   current_time: float,
                   duration: float,
                   precip_indices: tuple[int, ...],
                   forcing_schedule: ForcingSchedule | None = None,
                   forcing_states: list[torch.Tensor] | None = None):
    block0 = mesh.blocks[0]
    block_vars = mesh_vars[0]
    eos = block0.module("hydro.eos")
    thermo_y = block0.module("hydro.eos.thermo")
    intg = mesh.module("block0.intg")
    cycle = block0.cycle()
    block0.options.intg().tlim(current_time + duration)

    while not intg.stop(cycle, current_time):
        cycle += 1
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(mesh_vars)
        if forcing_schedule is not None and forcing_states is not None:
            refresh_time = next_forcing_time(forcing_schedule, len(forcing_states))
            if refresh_time is not None:
                dt = min(dt, max(refresh_time - current_time, 0.0))
        mesh.print_cycle_info(mesh_vars, current_time, dt)

        for stage in range(len(intg.stages)):
            mesh.forward(mesh_vars, dt, stage)
            block_vars = synchronize_hydro_state(mesh, block_vars)
            add_frigate_forcing(block_vars, eos, precip_indices, dt)
            block_vars = synchronize_hydro_state(mesh, block_vars)

        err = mesh.check_redo(mesh_vars)
        if err > 0:
            continue  # redo current step
        if err < 0:
            break  # terminate

        del_rho = evolve_kinetics(
            block_vars["hydro_w"], eos, thermo_x, thermo_y, kinet, dt
        )
        block_vars["hydro_u"][kICY:] += del_rho
        block_vars = synchronize_hydro_state(mesh, block_vars)

        current_time += dt
        mesh.make_outputs(mesh_vars, current_time)
        if forcing_schedule is not None and forcing_states is not None:
            refresh_time = next_forcing_time(forcing_schedule, len(forcing_states))
            if refresh_time is not None and current_time >= refresh_time - 1.0e-9:
                forcing_index = min(forcing_schedule.next_index, len(forcing_states) - 1)
                apply_ghost_zone_forcing(mesh, block_vars, forcing_states[forcing_index])
                forcing_schedule = ForcingSchedule(
                    interval_seconds=forcing_schedule.interval_seconds,
                    next_index=forcing_schedule.next_index + 1,
                )

    return mesh_vars, current_time

def nudge_from_ecmwf(mesh: Mesh,
                     thermo_x: ThermoX,
                     kinet: Kinetics,
                     block_vars: dict[str, torch.Tensor],
                     target_hydro_u: torch.Tensor,
                     precip_indices: tuple[int, ...],
                     index: int = 0,
                     refinement_level: int = 0):
    # downsampling to interpolated ECMWF resolution
    device = block_vars["hydro_u"].device
    nghost = mesh.blocks[0].options.coord().nghost()
    hydro_u = block_vars["hydro_u"].clone()

    for _ in range(refinement_level):
        u_int = hydro_u[..., nghost:-nghost, nghost:-nghost, :]
        u_int_coarsened = coarsen_spatial(u_int)
        dims = list(hydro_u.shape)
        dims[-2] = dims[-2] // 2 + nghost
        dims[-3] = dims[-3] // 2 + nghost
        hydro_u = torch.zeros(dims, dtype=hydro_u.dtype, device=device)
        hydro_u[..., nghost:-nghost, nghost:-nghost, :] = u_int_coarsened

    # read nudge data from ECMWF input
    du = target_hydro_u - hydro_u

    # upsample back to original resolution
    for _ in range(refinement_level):
        du_refined = refine_spatial(du, "area")
        du = du_refined[..., nghost:-nghost, nghost:-nghost, :]

    # nudge hydro_u
    block_vars["hydro_u"] += du

    # run hydrostatic adjustment for 600 seconds after nudging
    mesh.blocks[0].options.hydro().disable_flux_x2(True)
    mesh.blocks[0].options.hydro().disable_flux_x3(True)
    mesh_vars, _ = run_simulation(
        mesh, thermo_x, kinet, [block_vars], 0.0, 600.0, precip_indices
    )
    block_vars = mesh_vars[0]

    print_ok("Nudging from ECMWF data at index ", index, " completed.\n",
             "Min/Max density change = ", du[kIDN].min().item(), "/",
             du[kIDN].max().item(), '\n',
             "Min/Max pressure change = ", du[kIPR].min().item(), "/",
             du[kIPR].max().item(), '\n',
             "Min/Max vel1 change = ", du[kIV1].min().item(), "/",
             du[kIV1].max().item(), '\n',
             "Min/Max vel2 change = ", du[kIV2].min().item(), "/",
             du[kIV2].max().item(), '\n',
             "Min/Max vel3 change = ", du[kIV3].min().item(), "/",
             du[kIV3].max().item(), '\n',
             "Min/Max tracer change = ", du[kICY:].min().item(), "/",
             du[kICY:].max().item())

    mesh.blocks[0].options.hydro().disable_flux_x2(False)
    mesh.blocks[0].options.hydro().disable_flux_x3(False)

    return block_vars

def run_spinup(mesh: Mesh, 
               thermo_x: ThermoX,
               kinet: Kinetics,
               block_vars: dict[str, torch.Tensor],
               ecmwf_hydro_u: list[torch.Tensor],
               stage_topography: list[dict[str, torch.Tensor]],
               current_time: float,
               config_file: str,
               chunk_duration: float,
               precip_indices: tuple[int, ...]):
    for output in mesh.blocks[0].options.outputs():
        output.dt(3600.)

    for chunk in range(1, 4):
        start_clock = time.time()
        mesh_vars, current_time = run_simulation(
            mesh, thermo_x, kinet, [block_vars], current_time, chunk_duration, precip_indices
        )
        block_vars = mesh_vars[0]
        mesh, thermo_x, kinet, block_vars = refine_simulation(
            mesh, block_vars, stage_topography[chunk], config_file
        )

        block_vars = nudge_from_ecmwf(mesh, thermo_x, kinet,
                                      block_vars, ecmwf_hydro_u[chunk], precip_indices, chunk,
                                      refinement_level=chunk)
        end_clock = time.time()
        print_ok("Completed chunk ", chunk, "/ 3 for the day.\n",
                 "Current time = ", current_time, " seconds.\n",
                 "Elapsed time = ", end_clock - start_clock, " seconds.\n")
        print_ok("Refined variable shape = ", block_vars["hydro_w"].shape)

    return mesh, thermo_x, kinet, block_vars, current_time


def run_staged_ghost_schedule(
    mesh: Mesh,
    thermo_x: ThermoX,
    kinet: Kinetics,
    block_vars: dict[str, torch.Tensor],
    ecmwf_hydro_u: list[torch.Tensor],
    stage_topography: list[dict[str, torch.Tensor]],
    current_time: float,
    config_file: str,
    chunk_duration: float,
    precip_indices: tuple[int, ...],
    refine_at_18h: bool = False,
):
    if len(ecmwf_hydro_u) < 4:
        raise ValueError(
            f"Staged ghost forcing requires 4 ECMWF slices, found {len(ecmwf_hydro_u)}"
        )
    required_stage_count = 4 if refine_at_18h else 3
    if len(stage_topography) < required_stage_count:
        raise ValueError(
            "Staged ghost forcing requires topography stages through "
            f"{TOPO_SEQUENCE[required_stage_count - 1]}"
        )

    for output in mesh.blocks[0].options.outputs():
        output.dt(3600.)

    events = (
        (1, True, stage_topography[1]),
        (2, True, stage_topography[2]),
        (3, refine_at_18h, stage_topography[3] if refine_at_18h else None),
    )
    for index, should_refine, topo_vars in events:
        target_time = index * chunk_duration
        duration = max(target_time - current_time, 0.0)
        start_clock = time.time()
        mesh_vars, current_time = run_simulation(
            mesh, thermo_x, kinet, [block_vars], current_time, duration, precip_indices
        )
        block_vars = mesh_vars[0]

        if should_refine:
            mesh, thermo_x, kinet, block_vars = refine_simulation(
                mesh, block_vars, topo_vars, config_file
            )

        nghost = mesh.blocks[0].options.coord().nghost()
        target_hydro_u = refine_boundary_state_to_match(
            ecmwf_hydro_u[index], block_vars["hydro_u"].shape, nghost
        )
        block_vars = apply_ghost_zone_forcing(mesh, block_vars, target_hydro_u)
        end_clock = time.time()
        print_ok(
            "Completed staged ghost event ",
            index,
            "/ 3.\n",
            "Current time = ",
            current_time,
            " seconds.\n",
            "Elapsed time = ",
            end_clock - start_clock,
            " seconds.\n",
            "Variable shape = ",
            block_vars["hydro_w"].shape,
        )

    return mesh, thermo_x, kinet, block_vars, current_time


def refine_mesh(mesh: Mesh, device: torch.device, config_file: str) -> Mesh:
    old_block = mesh.blocks[0]
    file_numbers, next_times = [], []
    for out in old_block.get_outputs():
        file_numbers.append(out.file_number)
        next_times.append(out.next_time)

    coord = old_block.module("coord")
    nghost = old_block.options.coord().nghost()
    current_nx2 = max(coord.buffer("x2v").shape[0] - 2 * nghost, 1)
    current_nx3 = max(coord.buffer("x3v").shape[0] - 2 * nghost, 1)
    px = old_block.options.layout().px()
    py = old_block.options.layout().py()

    with open(config_file, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    base_nx2 = int(config["geometry"]["cells"]["nx2"])
    base_nx3 = int(config["geometry"]["cells"]["nx3"])
    next_nx2, next_nx3 = refined_global_horizontal_cells(current_nx2, current_nx3, px, py)
    config["geometry"]["cells"]["nx2"] = next_nx2
    config["geometry"]["cells"]["nx3"] = next_nx3
    next_nx2 = int(config["geometry"]["cells"]["nx2"])
    next_nx3 = int(config["geometry"]["cells"]["nx3"])
    refinement_factor = max(next_nx2 / max(base_nx2, 1), next_nx3 / max(base_nx3, 1))
    config["integration"]["implicit-scheme"] = implicit_scheme_for_refinement_factor(
        refinement_factor
    )

    refined_config = (
        Path(old_block.options.output_dir()) / f"{Path(config_file).stem}.refined.yaml"
    )
    refined_config.parent.mkdir(parents=True, exist_ok=True)
    refined_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    options = MeshOptions.from_yaml(str(refined_config), verbose=False)
    options.block().output_dir(old_block.options.output_dir())
    options.block().basename(old_block.options.basename())
    options.block().layout().backend(old_block.options.layout().backend())
    new_mesh = Mesh(options)
    new_mesh.to(device)
    for block in new_mesh.blocks:
        block.set_user_output_func(call_user_output)
    for n, out in enumerate(new_mesh.blocks[0].get_outputs()):
        out.file_number = file_numbers[n]
        out.next_time = next_times[n]
    return new_mesh


def refine_simulation(mesh: Mesh,
                      block_vars: dict[str, torch.Tensor],
                      topo_vars: dict[str, torch.Tensor],
                      config_file: str):
    device = block_vars["hydro_u"].device

    # refined meshblock
    mesh = refine_mesh(mesh, device, config_file)
    block = mesh.blocks[0]

    # thermo model
    thermo_y = block.module("hydro.eos.thermo")
    thermo_x = ThermoX(thermo_y.options)
    thermo_x.to(device)

    # kinetics model
    op_kinet = KineticsOptions.from_yaml(config_file)
    kinet = Kinetics(op_kinet)
    kinet.to(device)

    # refine variables
    nghost = block.options.coord().nghost()
    refined_hydro_u = conservative_refine(block_vars["hydro_u"], nghost)

    # topography
    topo_mask = (
        torch.meshgrid(
            block.module("coord").buffer("x3v"),
            block.module("coord").buffer("x2v"),
            block.module("coord").buffer("x1v"),
            indexing="ij",
        )[2]
        < build_topography_field_for_block(block, topo_vars)
    )

    # eos model
    eos = block.module("hydro.eos")
    hydro_w = eos.compute("U->W", (refined_hydro_u,))
    mesh_vars, _ = mesh.initialize([{"hydro_w": hydro_w}])
    mesh_vars[0]["topo"] = topo_mask
    block_vars = mesh_vars[0]

    return mesh, thermo_x, kinet, block_vars


def prepare_initial_state(
    mesh: Mesh,
    thermo_x: ThermoX,
    block_vars: dict[str, torch.Tensor],
    topo_vars: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], float]:
    block = mesh.blocks[0]
    block_vars = equilibrate_initial_fields(block, thermo_x, block_vars)
    x1v = torch.meshgrid(
        block.module("coord").buffer("x3v"),
        block.module("coord").buffer("x2v"),
        block.module("coord").buffer("x1v"),
        indexing="ij",
    )[2]
    topo0 = build_topography_field_for_block(block, topo_vars)
    mesh_vars, current_time = mesh.initialize([{"hydro_w": block_vars["hydro_w"]}])
    block_vars = mesh_vars[0]
    block_vars["topo"] = x1v < topo0
    return block_vars, current_time


def log_loaded_inputs(
    block_vars: dict[str, torch.Tensor],
    stage_topography: list[dict[str, torch.Tensor]],
) -> None:
    print_ok("Block variables loaded from ECMWF input:")
    for key, data in block_vars.items():
        print(f"  - {key}: shape = {data.shape}, dtype = {data.dtype}, device = {data.device}")

    print_ok("Topography stages:")
    labels = TOPO_SEQUENCE if len(stage_topography) == len(TOPO_SEQUENCE) else (TOPO_SEQUENCE[0],)
    for label, topo_vars in zip(labels, stage_topography):
        print(f"  - {label}: raw shape = {tuple(topo_vars['topography'].shape)}")


def build_forecast_context(args: argparse.Namespace) -> ForecastContext:
    restart_file = resolve_restart_file(args.input_dir)
    topography_root = (
        Path(args.topography_dir).resolve()
        if args.topography_dir is not None
        else default_topography_dir(args.config)
    )
    topography_files = discover_topography_files(
        str(topography_root),
        required_labels=selected_topography_labels(args.refinement_mode, args.topography_stage),
    )
    precip_indices = precipitation_tracer_indices(args.config)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mesh, thermo_x, kinet, device = create_mesh(
        args.config, args.output_dir, requested_device=args.device
    )
    time_slices, topography_modules = load_time_series_inputs(
        restart_file, topography_files, device
    )
    block_vars = time_slices[0]
    block = mesh.blocks[0]
    eos = block.module("hydro.eos")
    ecmwf_hydro_u = [eos.compute("W->U", (slice_vars["hydro_w"],)) for slice_vars in time_slices]
    if args.refinement_mode == "staged":
        stage_topography = [topography_modules[label] for label in TOPO_SEQUENCE]
    else:
        stage_topography = [topography_modules[args.topography_stage]]
    log_loaded_inputs(block_vars, stage_topography)
    block_vars, current_time = prepare_initial_state(
        mesh, thermo_x, block_vars, stage_topography[0]
    )

    return ForecastContext(
        mesh=mesh,
        thermo_x=thermo_x,
        kinet=kinet,
        device=device,
        block_vars=block_vars,
        ecmwf_hydro_u=ecmwf_hydro_u,
        stage_topography=stage_topography,
        precip_indices=precip_indices,
        current_time=current_time,
        refine_at_18h=args.refine_at_18h,
    )


def with_disabled_lateral_fluxes(block: MeshBlock):
    block.options.hydro().disable_flux_x2(True)
    block.options.hydro().disable_flux_x3(True)


def with_enabled_lateral_fluxes(block: MeshBlock):
    block.options.hydro().disable_flux_x2(False)
    block.options.hydro().disable_flux_x3(False)


def run_hydrostatic_adjustment(ctx: ForecastContext, duration: float) -> None:
    block = ctx.mesh.blocks[0]
    start_clock = time.time()
    with_disabled_lateral_fluxes(block)
    ctx.mesh.make_outputs([ctx.block_vars], ctx.current_time)
    mesh_vars, ctx.current_time = run_simulation(
        ctx.mesh,
        ctx.thermo_x,
        ctx.kinet,
        [ctx.block_vars],
        ctx.current_time,
        duration,
        ctx.precip_indices,
    )
    ctx.block_vars = mesh_vars[0]
    ctx.current_time = 0.0
    with_enabled_lateral_fluxes(block)
    end_clock = time.time()
    print_ok(
        "Step 1 (hydrostatic adjustment) completed.\n",
        "Current time = ", ctx.current_time, " seconds.\n",
        "Current cycle = ", ctx.mesh.blocks[0].cycle(), ".\n",
        "Elapsed time = ", end_clock - start_clock, " seconds.\n",
    )


def run_spinup_stage(ctx: ForecastContext, config_file: str, chunk_duration: float) -> None:
    start_clock = time.time()
    (
        ctx.mesh,
        ctx.thermo_x,
        ctx.kinet,
        ctx.block_vars,
        ctx.current_time,
    ) = run_spinup(
        ctx.mesh,
        ctx.thermo_x,
        ctx.kinet,
        ctx.block_vars,
        ctx.ecmwf_hydro_u,
        ctx.stage_topography,
        ctx.current_time,
        config_file,
        chunk_duration,
        ctx.precip_indices,
    )
    end_clock = time.time()
    print_ok(
        "Step 2 (spinup) completed.\n",
        "Current time = ", ctx.current_time, " seconds.\n",
        "Current cycle = ", ctx.mesh.blocks[0].cycle(), ".\n",
        "Elapsed time = ", end_clock - start_clock, " seconds.\n",
    )


def run_spinup_stage_with_ghost_forcing(
    ctx: ForecastContext, config_file: str, chunk_duration: float
) -> None:
    start_clock = time.time()
    (
        ctx.mesh,
        ctx.thermo_x,
        ctx.kinet,
        ctx.block_vars,
        ctx.current_time,
    ) = run_staged_ghost_schedule(
        ctx.mesh,
        ctx.thermo_x,
        ctx.kinet,
        ctx.block_vars,
        ctx.ecmwf_hydro_u,
        ctx.stage_topography,
        ctx.current_time,
        config_file,
        chunk_duration,
        ctx.precip_indices,
        refine_at_18h=ctx.refine_at_18h,
    )
    end_clock = time.time()
    print_ok(
        "Step 2 (staged refinement with ghost forcing) completed.\n",
        "Current time = ", ctx.current_time, " seconds.\n",
        "Current cycle = ", ctx.mesh.blocks[0].cycle(), ".\n",
        "Elapsed time = ", end_clock - start_clock, " seconds.\n",
    )


def run_prediction_stage(ctx: ForecastContext, duration: float) -> None:
    start_clock = time.time()
    mesh_vars, ctx.current_time = run_simulation(
        ctx.mesh,
        ctx.thermo_x,
        ctx.kinet,
        [ctx.block_vars],
        ctx.current_time,
        duration,
        ctx.precip_indices,
    )
    ctx.block_vars = mesh_vars[0]
    end_clock = time.time()
    print_ok(
        "Step 3 (prediction) completed.\n",
        "Current time = ", ctx.current_time, " seconds.\n",
        "Current cycle = ", ctx.mesh.blocks[0].cycle(), ".\n",
        "Elapsed time = ", end_clock - start_clock, " seconds.\n",
    )


def run_prediction_stage_with_ghost_forcing(
    ctx: ForecastContext,
    duration: float,
    forcing_interval_seconds: float,
) -> None:
    start_clock = time.time()
    mesh_vars, ctx.current_time = run_simulation(
        ctx.mesh,
        ctx.thermo_x,
        ctx.kinet,
        [ctx.block_vars],
        ctx.current_time,
        duration,
        ctx.precip_indices,
        forcing_schedule=ForcingSchedule(interval_seconds=forcing_interval_seconds),
        forcing_states=ctx.ecmwf_hydro_u,
    )
    ctx.block_vars = mesh_vars[0]
    end_clock = time.time()
    print_ok(
        "Step 2 (low-resolution prediction with ghost forcing) completed.\n",
        "Current time = ", ctx.current_time, " seconds.\n",
        "Current cycle = ", ctx.mesh.blocks[0].cycle(), ".\n",
        "Elapsed time = ", end_clock - start_clock, " seconds.\n",
    )


def finalize_forecast(ctx: ForecastContext) -> None:
    ctx.mesh.finalize([ctx.block_vars], ctx.current_time)
    print_ok("Simulation finalized at time = ", ctx.current_time, " seconds.")


def run_forecast(args: argparse.Namespace) -> None:
    ctx = None
    try:
        ctx = build_forecast_context(args)
        run_hydrostatic_adjustment(ctx, args.hydrostatic_duration)
        if args.refinement_mode == "staged":
            if args.forcing_mode == "ghost":
                run_spinup_stage_with_ghost_forcing(
                    ctx, args.config, args.spinup_chunk_duration
                )
            else:
                run_spinup_stage(ctx, args.config, args.spinup_chunk_duration)
            run_prediction_stage(ctx, args.prediction_duration)
        elif args.forcing_mode == "ghost":
            run_prediction_stage_with_ghost_forcing(
                ctx, args.prediction_duration, args.forcing_interval_seconds
            )
        else:
            run_prediction_stage(ctx, args.prediction_duration)
        finalize_forecast(ctx)
    finally:
        if ctx is not None:
            try:
                ctx.mesh.finalize([ctx.block_vars], ctx.current_time)
            except Exception:
                pass
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Run hydrodynamic simulation.")
    parser.add_argument(
        "-c", "--config", type=str,
        required=True, help="YAML configuration file."
    )
    parser.add_argument(
        "-i", "--input_dir", type=str,
        default=None, help="directory containing regridded ERA5 tensor .part files."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str,
        default=None, help="forecast output directory."
    )
    parser.add_argument(
        "--topography-dir", type=str,
        default=None, help="directory containing *_topo_<resolution>.pt files."
    )
    parser.add_argument(
        "--hydrostatic-duration", type=float,
        default=600.0, help="duration in seconds for the initial hydrostatic adjustment."
    )
    parser.add_argument(
        "--spinup-chunk-duration", type=float,
        default=21600.0, help="duration in seconds for each ECMWF-synced spinup chunk."
    )
    parser.add_argument(
        "--refinement-mode", type=str,
        default="staged", choices=["staged", "none"],
        help="use staged refinement or stay on a single mesh for the full run."
    )
    parser.add_argument(
        "--forcing-mode", type=str,
        default="nudge", choices=["nudge", "ghost"],
        help=(
            "apply whole-domain nudging or refresh lateral ghost zones from ERA5. "
            "With staged refinement, ghost mode refines at 06h and 12h and applies "
            "a ghost-only refresh at 18h by default; pass --refine-at-18h to refine "
            "onto 0p3km at 18h instead."
        )
    )
    parser.add_argument(
        "--forcing-interval-seconds", type=float,
        default=21600.0, help="interval between ERA5 boundary updates in seconds."
    )
    parser.add_argument(
        "--refine-at-18h",
        action="store_true",
        help=(
            "with staged ghost forcing, refine again at 18h onto the 0p3km stage "
            "instead of applying only a ghost-zone refresh."
        ),
    )
    parser.add_argument(
        "--topography-stage", type=str,
        default=None, choices=list(TOPO_SEQUENCE),
        help="topography label to use when refinement is disabled."
    )
    parser.add_argument(
        "--prediction-duration", type=float,
        default=7 * 86400.0, help="duration in seconds for the final forecast segment."
    )
    parser.add_argument(
        "--device", type=str,
        default="auto", choices=["auto", "cpu", "cuda"],
        help="execution device override."
    )
    args = parser.parse_args()

    if args.input_dir is None:
        args.input_dir = str(default_input_dir(args.config))
    if args.output_dir is None:
        args.output_dir = str(default_output_dir(args.config))
    required_topography = selected_topography_labels(args.refinement_mode, args.topography_stage)
    args.topography_stage = required_topography[0]

    run_forecast(args)

if __name__ == "__main__":
    main()
