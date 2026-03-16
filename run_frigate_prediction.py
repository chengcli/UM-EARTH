import argparse
import time
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


def implicit_scheme_for_refinement_factor(refinement_factor: float) -> int:
    return 1

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
        if candidate.suffix != ".part":
            raise FileNotFoundError(f"Expected a .part restart tensor, found {candidate}")
        return candidate

    if not candidate.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {candidate}")

    preferred = candidate / "input.part"
    if preferred.exists():
        return preferred

    matches = sorted(candidate.glob("*.part"))
    if not matches:
        raise FileNotFoundError(f"Could not find any .part restart tensor in {candidate}")
    if len(matches) > 1:
        print_ok("Multiple restart tensors found; using", matches[0].name)
    return matches[0]


def parse_topography_label(path: Path) -> str:
    marker = "_topo_"
    if marker not in path.stem:
        raise ValueError(f"Topography tensor does not follow '*_topo_<label>.pt': {path}")
    return path.stem.split(marker, 1)[1]


def discover_topography_files(topography_dir: str) -> dict[str, Path]:
    topo_root = Path(topography_dir).resolve()
    topo_paths = sorted(topo_root.glob("*_topo_*.pt"))
    if not topo_paths:
        raise FileNotFoundError(f"Could not find any topography tensors in {topo_root}")

    discovered = {parse_topography_label(path): path for path in topo_paths}
    mapping = {label: discovered[label] for label in TOPO_SEQUENCE if label in discovered}
    missing = [label for label in TOPO_SEQUENCE if label not in mapping]
    if missing:
        raise FileNotFoundError(
            f"Missing topography tensors for {missing} in {topo_root}; found {sorted(mapping)}"
        )
    return mapping


def default_topography_dir(config_file: str) -> Path:
    return infer_run_dir(config_file) / "topography" / "products"


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


def create_mesh(config_file: str, output_dir: str, requested_device: str = "auto"):
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
    module = torch.jit.load(str(restart_file))
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


def build_topography_field_for_block(
    block: MeshBlock,
    topo_vars: dict[str, torch.Tensor],
) -> torch.Tensor:
    coord = block.module("coord")
    nghost = block.options.coord().nghost()
    nc2 = coord.buffer("x2v").shape[0]
    nc3 = coord.buffer("x3v").shape[0]
    return build_topography_field(
        topo_vars,
        (max(nc2 - 2 * nghost, 1), max(nc3 - 2 * nghost, 1)),
        nghost,
    )

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

def add_frigate_forcing(block_vars: dict[str, torch.Tensor],
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

def run_simulation(mesh: Mesh,
                   thermo_x: ThermoX,
                   kinet: Kinetics,
                   mesh_vars: list[dict[str, torch.Tensor]],
                   current_time: float,
                   duration: float):
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
        mesh.print_cycle_info(mesh_vars, current_time, dt)

        for stage in range(len(intg.stages)):
            mesh.forward(mesh_vars, dt, stage)
            add_frigate_forcing(block_vars, dt)

        err = mesh.check_redo(mesh_vars)
        if err > 0:
            continue  # redo current step
        if err < 0:
            break  # terminate

        del_rho = evolve_kinetics(
            block_vars["hydro_w"], eos, thermo_x, thermo_y, kinet, dt
        )
        block_vars["hydro_u"][kICY:] += del_rho

        current_time += dt
        mesh.make_outputs(mesh_vars, current_time)

    return mesh_vars, current_time

def nudge_from_ecmwf(mesh: Mesh,
                     thermo_x: ThermoX,
                     kinet: Kinetics,
                     block_vars: dict[str, torch.Tensor],
                     target_hydro_u: torch.Tensor,
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
    mesh_vars, _ = run_simulation(mesh, thermo_x, kinet, [block_vars], 0.0, 600.0)
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
               chunk_duration: float):
    for output in mesh.blocks[0].options.outputs():
        output.dt(3600.)

    for chunk in range(1, 4):
        start_clock = time.time()
        mesh_vars, current_time = run_simulation(
            mesh, thermo_x, kinet, [block_vars], current_time, chunk_duration
        )
        block_vars = mesh_vars[0]
        mesh, thermo_x, kinet, block_vars = refine_simulation(
            mesh, block_vars, stage_topography[chunk], config_file
        )

        block_vars = nudge_from_ecmwf(mesh, thermo_x, kinet,
                                      block_vars, ecmwf_hydro_u[chunk], chunk,
                                      refinement_level=chunk)
        end_clock = time.time()
        print_ok("Completed chunk ", chunk, "/ 3 for the day.\n",
                 "Current time = ", current_time, " seconds.\n",
                 "Elapsed time = ", end_clock - start_clock, " seconds.\n")
        print_ok("Refined variable shape = ", block_vars["hydro_w"].shape)

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

    with open(config_file, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    base_nx2 = int(config["geometry"]["cells"]["nx2"])
    base_nx3 = int(config["geometry"]["cells"]["nx3"])
    config["geometry"]["cells"]["nx2"] = current_nx2 * 2 if current_nx2 > 1 else 1
    config["geometry"]["cells"]["nx3"] = current_nx3 * 2 if current_nx3 > 1 else 1
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

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Run hydrodynamic simulation.")
    parser.add_argument(
        "-c", "--config", type=str,
        required=True, help="YAML configuration file."
    )
    parser.add_argument(
        "-i", "--input_dir", type=str,
        default='./input', help="input directory."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str,
        default='./output', help="output directory."
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
        "--prediction-duration", type=float,
        default=7 * 86400.0, help="duration in seconds for the final forecast segment."
    )
    parser.add_argument(
        "--device", type=str,
        default="auto", choices=["auto", "cpu", "cuda"],
        help="execution device override."
    )
    args = parser.parse_args()

    restart_file = resolve_restart_file(args.input_dir)
    topography_root = (
        Path(args.topography_dir).resolve()
        if args.topography_dir is not None
        else default_topography_dir(args.config)
    )
    topography_files = discover_topography_files(str(topography_root))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mesh = None
    block_vars = None
    current_time = 0.0
    try:
        mesh, thermo_x, kinet, device = create_mesh(
            args.config, args.output_dir, requested_device=args.device
        )
        time_slices, topography_modules = load_time_series_inputs(
            restart_file, topography_files, device
        )
        block_vars = time_slices[0]
        print_ok("Block variables loaded from ECMWF input:")
        for key, data in block_vars.items():
            print(f"  - {key}: shape = {data.shape}, dtype = {data.dtype}, device = {data.device}")

        block = mesh.blocks[0]
        eos = block.module("hydro.eos")
        ecmwf_hydro_u = [eos.compute("W->U", (slice_vars["hydro_w"],)) for slice_vars in time_slices]

        stage_topography = [topography_modules[label] for label in TOPO_SEQUENCE]

        print_ok("Topography stages:")
        for label, topo_vars in zip(TOPO_SEQUENCE, stage_topography):
            print(f"  - {label}: raw shape = {tuple(topo_vars['topography'].shape)}")

        # initialize model
        block_vars = equilibrate_initial_fields(block, thermo_x, block_vars)
        x1v = torch.meshgrid(
            block.module("coord").buffer("x3v"),
            block.module("coord").buffer("x2v"),
            block.module("coord").buffer("x1v"),
            indexing="ij",
        )[2]
        topo0 = build_topography_field_for_block(block, stage_topography[0])
        mesh_vars, current_time = mesh.initialize([{"hydro_w": block_vars["hydro_w"]}])
        block_vars = mesh_vars[0]
        block_vars["topo"] = x1v < topo0

        # Step 1
        start_clock = time.time()
        block.options.hydro().disable_flux_x2(True)
        block.options.hydro().disable_flux_x3(True)
        mesh.make_outputs([block_vars], current_time)
        mesh_vars, current_time = run_simulation(
            mesh, thermo_x, kinet, [block_vars], current_time, args.hydrostatic_duration
        )
        block_vars = mesh_vars[0]
        current_time = 0.0
        block.options.hydro().disable_flux_x2(False)
        block.options.hydro().disable_flux_x3(False)
        end_clock = time.time()
        print_ok("Step 1 (hydrostatic adjustment) completed.\n",
                 "Current time = ", current_time, " seconds.\n",
                 "Current cycle = ", mesh.blocks[0].cycle(), ".\n",
                 "Elapsed time = ", end_clock - start_clock, " seconds.\n")

        # Step 2
        start_clock = end_clock
        mesh, thermo_x, kinet, block_vars, current_time = run_spinup(
            mesh, thermo_x, kinet, block_vars, ecmwf_hydro_u,
            stage_topography, current_time, args.config, args.spinup_chunk_duration
        )
        end_clock = time.time()
        print_ok("Step 2 (spinup) completed.\n",
                 "Current time = ", current_time, " seconds.\n",
                 "Current cycle = ", mesh.blocks[0].cycle(), ".\n",
                 "Elapsed time = ", end_clock - start_clock, " seconds.\n")

        # Step 3
        start_clock = end_clock
        mesh_vars, current_time = run_simulation(
            mesh, thermo_x, kinet, [block_vars], current_time, args.prediction_duration
        )
        block_vars = mesh_vars[0]
        end_clock = time.time()
        print_ok("Step 3 (prediction) completed.\n",
                 "Current time = ", current_time, " seconds.\n",
                 "Current cycle = ", mesh.blocks[0].cycle(), ".\n",
                 "Elapsed time = ", end_clock - start_clock, " seconds.\n")

        # Step 4
        mesh.finalize([block_vars], current_time)
        print_ok("Simulation finalized at time = ", current_time, " seconds.")
    finally:
        if mesh is not None and block_vars is not None:
            try:
                mesh.finalize([block_vars], current_time)
            except Exception:
                pass
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
