import torch
import yaml
import argparse
import time
from snapy import MeshBlockOptions, MeshBlock
from snapy import kIDN, kIPR, kICY, kIV1, kIV2, kIV3
from kintera import ThermoX, KineticsOptions, Kinetics
from refine_utils import(
        conservative_refine, 
        refine_meshblock,
        refine_spatial,
        coarsen_spatial
        )
from paddle import evolve_kinetics

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

def create_block(config_file: str):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # set hydrodynamic options
    op = MeshBlockOptions.from_yaml(config_file)
    block = MeshBlock(op)

    # use cuda if available
    if torch.cuda.is_available() and op.layout().backend() == "nccl":
        device = torch.device(block.device())
        print("Using device = ", device)
    else:
        device = torch.device("cpu")

    block.to(device)
    block.set_user_output_func(call_user_output)

    # thermo model
    thermo_y = block.module("hydro.eos.thermo")
    thermo_x = ThermoX(thermo_y.options)
    thermo_x.to(device)

    # kinetics model
    op_kinet = KineticsOptions.from_yaml(config_file)
    kinet = Kinetics(op_kinet)
    kinet.to(device)

    return block, thermo_x, kinet, device

def load_ecmwf_input(input_file: str, index: int = 0,
                     device: torch.device = torch.device("cpu")):
    module = torch.jit.load(input_file)
    block_vars = {}
    for name, data in module.named_buffers(recurse=True):
        block_vars[name] = data[index].to(device).to(torch.double)

    for name, data in module.named_parameters(recurse=True):
        block_vars[name] = data[index].to(device).to(torch.double)
    return block_vars

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

def run_simulation(block: MeshBlock, 
                   thermo_x: ThermoX,
                   kinet: Kinetics,
                   block_vars: dict[str, torch.Tensor],
                   current_time: float,
                   duration: float):
    eos = block.module("hydro.eos")
    thermo_y = block.module("hydro.eos.thermo")
    block.options.intg().tlim(current_time + duration)

    while not block.intg.stop(block.inc_cycle(), current_time):
        dt = block.max_time_step(block_vars)
        block.print_cycle_info(block_vars, current_time, dt)

        for stage in range(len(block.intg.stages)):
            block.forward(block_vars, dt, stage)

        err = block.check_redo(block_vars)
        if err > 0:
            continue  # redo current step
        if err < 0:
            break  # terminate

        del_rho = evolve_kinetics(
            block_vars["hydro_w"], eos, thermo_x, thermo_y, kinet, dt
        )
        block_vars["hydro_u"][kICY:] += del_rho

        current_time += dt
        block.make_outputs(block_vars, current_time)

    return block_vars, current_time

def nudge_from_ecmwf(block: MeshBlock,
                     block_vars: dict[str, torch.Tensor],
                     input_file: str,
                     index: int = 0):
    # downsampling to interpolated ECMWF resolution
    device = block_vars["hydro_u"].device
    nghost = block.options.coord().nghost()
    hydro_u = block_vars["hydro_u"].clone()

    for n in range(index):
        u_int = hydro_u[..., nghost:-nghost, nghost:-nghost, :]
        u_int_coarsened = coarsen_spatial(u_int)
        dims = list(hydro_u.shape)
        dims[-2] = (dims[-2] - 2 * nghost) // 2 + 2 * nghost
        dims[-3] = (dims[-3] - 2 * nghost) // 2 + 2 * nghost
        hydro_u = torch.zeros(dims, dtype=hydro_u.dtype, device=device)
        hydro_u[..., nghost:-nghost, nghost:-nghost, :] = u_int_coarsened

    # read nudge data from ECMWF input
    nudge_vars = load_ecmwf_input(input_file, index=index, device=device)
    eos = block.module("hydro.eos")
    du = eos.compute("W->U", (nudge_vars["hydro_w"],)) - hydro_u

    # upsample back to original resolution
    for n in range(index):
        du_int = du[..., nghost:-nghost, nghost:-nghost, :]
        du_int_refined = refine_spatial(u_int, "area")
        dims = list(du.shape)
        dims[-2] = (dims[-2] - 2 * nghost) * 2 + 2 * nghost
        dims[-3] = (dims[-3] - 2 * nghost) * 2 + 2 * nghost
        du = torch.zeros(dims, dtype=hydro_u.dtype, device=device)
        du[..., nghost:-nghost, nghost:-nghost, :] = du_int_refined

    # nudge hydro_u
    alpha = 0.9
    hydro_u = block_vars["hydro_u"]
    hydro_u[kIDN] += du[kIDN] * alpha
    hydro_u[kIPR] += du[kIPR] * alpha
    hydro_u[kIV2] += du[kIV2] * alpha
    hydro_u[kIV3] += du[kIV3] * alpha

    for n in range(kICY, hydro_u.shape[0]):
        hydro_u[n] += du[n] * alpha

    return block_vars

def run_spinup(block: MeshBlock, 
               thermo_x: ThermoX,
               kinet: Kinetics,
               block_vars: dict[str, torch.Tensor],
               current_time: float,
               config_file: str,
               input_file: str):
    for output in block.options.outputs():
        output.dt(3600.)

    duration = 2160.  # seconds (6 hours)
    for chunk in range(3):
        start_clock = time.time()
        block_vars, current_time = run_simulation(block, thermo_x, kinet,
                                                  block_vars, current_time, duration)
        block, thermo_x, kinet, block_vars = refine_simulation(
                block, block_vars, config_file)
        end_clock = time.time()

        #block_vars = nudge_from_ecmwf(block, block_vars, input_file, chunk + 1)
        print_ok("Completed chunk ", chunk+1, "/ 3 for the day.\n",
                 "Elapsed time = ", end_clock - start_clock, " seconds.\n")
        print_ok("Refined variable shape = ", block_vars["hydro_w"].shape)

    return block, thermo_x, kinet, block_vars, current_time

def refine_simulation(block: MeshBlock,
                      block_vars: dict[str, torch.Tensor],
                      config_file: str):
    device = block_vars["hydro_u"].device

    # refined meshblock
    block = refine_meshblock(block)
    block.to(device)
    block.set_user_output_func(call_user_output)

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
    block_vars["hydro_u"] = conservative_refine(block_vars["hydro_u"], nghost)

    # eos model
    eos = block.module("hydro.eos")
    block_vars["hydro_w"] = eos.compute("U->W", (block_vars["hydro_u"],))
    block_vars, _ = block.initialize(block_vars)

    return block, thermo_x, kinet, block_vars

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Run hydrodynamic simulation.")
    parser.add_argument(
        "-c", "--config", type=str,
        required=True, help="YAML configuration file."
    )
    parser.add_argument(
        "-i", "--input", type=str, 
        required=True, help="Initial input file.",
        default=""
    )
    args = parser.parse_args()

    block, thermo_x, kinet, device = create_block(args.config)
    block_vars = load_ecmwf_input(args.input, index=0, device=device)
    for key, data in block_vars.items():
        print(f"{key}: shape = {data.shape}, dtype = {data.dtype}, device = {data.device}")

    block_vars = equilibrate_initial_fields(block, thermo_x, block_vars)
    block_vars, current_time = block.initialize(block_vars)

    ### Step 1: Run hydrostatic adjustment for 1200 seconds ###
    start_clock = time.time()

    block.options.hydro().disable_flux_x2(True)
    block.options.hydro().disable_flux_x3(True)
    block.options.hydro().icorr().scheme(1)
    #block.options.intg().cfl(0.45)

    block.make_outputs(block_vars, current_time)
    for output in block.options.outputs():
        output.dt(600.)

    block_vars, current_time = run_simulation(block, thermo_x, kinet,
                                              block_vars, current_time, 1200.)

    current_time = 0.
    for out in block.get_outputs():
        out.next_time = 3600.

    end_clock = time.time()
    print_ok("Step 1 (hydrostatic adjustment) completed.\n",
          "Current time = ", current_time, " seconds.\n",
          "Current cycle = ", block.cycle(), ".\n",
          "Elapsed time = ", end_clock - start_clock, " seconds.\n")

    ### Step 2: Run Spin-up for 1 day with increasing resolution ###
    start_clock = end_clock
    block.options.hydro().disable_flux_x2(False)
    block.options.hydro().disable_flux_x3(False)
    #block.options.hydro().icorr().scheme(0)
    #block.options.intg().cfl(0.45)

    block, thermo_x, kinet, block_vars, current_time = run_spinup(
            block, thermo_x, kinet, block_vars, current_time, 
            args.config, args.input)

    end_clock = time.time()
    print_ok("Step 2 (spinup) completed.\n",
          "Current time = ", current_time, " seconds.\n",
          "Current cycle = ", block.cycle(), ".\n",
          "Elapsed time = ", end_clock - start_clock, " seconds.\n")

    ### Step 3: Run prediction for the next 32 hours ###
    start_clock = end_clock
    duration = 32 * 3600.  # seconds 
    block_vars, current_time = run_simulation(block, thermo_x, kinet,
                                              block_vars, current_time, duration)

    end_clock = time.time()
    print_ok("Step 3 (prediction) completed.\n",
          "Current time = ", current_time, " seconds.\n",
          "Current cycle = ", block.cycle(), ".\n",
          "Elapsed time = ", end_clock - start_clock, " seconds.\n")

    ### Step 4: Clean up and finalize ###
    block.finalize(block_vars, current_time)
    print_ok("Simulation finalized at time = ", current_time, " seconds.")

if __name__ == "__main__":
    main()
