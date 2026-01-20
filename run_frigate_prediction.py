import torch
import yaml
import argparse
from snapy import MeshBlockOptions, MeshBlock
from snapy import kIDN, kIPR, kICY, kIV1, kIV2, kIV3
from kintera import ThermoX, KineticsOptions, Kinetics
from refine_utils import conservative_refine, refine_meshblock
from paddle import (
    setup_profile,
    evolve_kinetics,
)

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
    print('device = ', device)
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

def nudge_from_ecmwf(block_vars: dict[str, torch.Tensor], input_file: str):
    return block_vars

def run_simulation_one_day(block: MeshBlock, 
                           thermo_x: ThermoX,
                           kinet: Kinetics,
                           block_vars: dict[str, torch.Tensor],
                           current_time: float):
    block.options.hydro().disalbe_flux_x2(False)
    block.options.hydro().disalbe_flux_x3(False)
    block.options.hydro().icoor().scheme(0)

    for output in block.options.outputs():
        output.dt(3600.)

    for chunk in range(4):
        block_vars, current_time = run_simulation(block, thermo_x, kinet,
                                                  block_vars, current_time, 21600.)
        block_vars = nudge_from_ecmwf(block_vars, args.input)

    return block_vars, current_time

def refine_simulation(block: MeshBlock,
                      block_vars: dict[str, torch.Tensor],
                      device: torch.device = torch.device("cpu")):
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
    block_vars["hydro_u"] = conservative_refine(block_vars["hydro_u"])

    # eos model
    eos = block.module("hydro.eos")
    block_vars["hydro_w"] = eos.compute("U->W", {block_vars["hydro_u"]})
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
    hydro_w = block_vars["hydro_w"]
    print(hydro_w[kIDN].min(), hydro_w[kIDN].max())
    print(hydro_w[kICY:].min(), hydro_w[kICY:].max())

    block_vars, current_time = block.initialize(block_vars)

    ### Step 1: Run hydrostatic adjustment for 2400 seconds ###
    block.options.hydro().disable_flux_x2(True)
    block.options.hydro().disable_flux_x3(True)
    block.options.hydro().icorr().scheme(9)
    block.options.intg().cfl(0.05)

    block.make_outputs(block_vars, current_time)
    block_vars, current_time = run_simulation(block, thermo_x, kinet,
                                              block_vars, current_time, 2400.)

    current_time = 0.
    print("[OK] Step 1 (hydrostatic adjustment) completed.\n"
          "Current time = ", current_time, " seconds.\n"
          "Current cycle = ", block.cycle(), ".\n")

    exit(0)

    ### Step 2: Run simulation for 7 days with increasing resolution ###
    days = 7
    for day in range(days):
        block_vars, current_time = run_simulation_one_day(block, thermo_x, kinet,
                                                          block_vars, current_time)
        block, thermo_x, kinet, block_vars = refine_simulation(block, block_vars, device)
        print("[OK] Day ", day+1, " completed.\n"
              "Current time = ", current_time, " seconds.\n")
    print("[OK] Step 2 (bootstrap) completed.\n"
          "Current time = ", current_time, " seconds.\n"
          "Current cycle = ", block.cycle(), ".\n")

    ### Step 3: Run prediction for next week ###
    duration = 604800.  # 7 days in seconds 
    block_vars, current_time = run_simulation(block, thermo_x, kinet,
                                              block_vars, current_time, duration)
    print("[OK] Step 3 (prediction) completed.\n"
          "Current time = ", current_time, " seconds.\n"
          "Current cycle = ", block.cycle(), ".\n")

    ### Step 4: Clean up and finalize ###
    block.finalize(block_vars, current_time)
    print("[OK] Simulation finalized at time = ", current_time, " seconds.")

if __name__ == "__main__":
    main()
