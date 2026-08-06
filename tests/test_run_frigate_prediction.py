from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "run_frigate_prediction.py"


def load_module():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location("run_frigate_prediction", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_restart_file_prefers_existing_part(tmp_path):
    module = load_module()
    tensor_dir = tmp_path / "tensors"
    tensor_dir.mkdir()
    preferred = tensor_dir / "regridded_block_0_0.part"
    preferred.write_text("", encoding="utf-8")

    assert module.resolve_restart_file(str(tensor_dir)) == preferred
    assert module.resolve_restart_file(str(preferred)) == preferred


def test_default_input_dir_finds_regridded_tensor_dir(tmp_path):
    module = load_module()
    run_dir = tmp_path / "run"
    tensor_dir = run_dir / "era5" / "bounds" / "regridded_case_tensors"
    tensor_dir.mkdir(parents=True)
    config_file = run_dir / "case.yaml"
    config_file.write_text("", encoding="utf-8")

    assert module.default_input_dir(str(config_file)) == tensor_dir


def test_default_input_dir_finds_forecast_tensor_dir(tmp_path):
    module = load_module()
    run_dir = tmp_path / "run"
    tensor_dir = run_dir / "forecast_input" / "regridded_case_tensors"
    tensor_dir.mkdir(parents=True)
    config_file = run_dir / "case.yaml"
    config_file.write_text("", encoding="utf-8")

    assert module.default_input_dir(str(config_file)) == tensor_dir


def test_default_output_dir_uses_run_local_forecast_output(tmp_path):
    module = load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_file = run_dir / "case.yaml"
    config_file.write_text("", encoding="utf-8")

    assert module.default_output_dir(str(config_file)) == run_dir / "forecast_output"


def test_is_runtime_restart_state_detects_runtime_restart_metadata():
    module = load_module()
    state = {
        "hydro_u": torch.zeros((8, 62, 29, 56), dtype=torch.float64),
        "hydro_w": torch.zeros((8, 62, 29, 56), dtype=torch.float64),
        "last_time": torch.tensor([86400.0], dtype=torch.float64),
        "last_cycle": torch.tensor([123], dtype=torch.int64),
    }

    assert module.is_runtime_restart_state(state) is True
    assert module.is_runtime_restart_state({"hydro_w": torch.zeros((4, 8, 62, 29, 56))}) is False


def test_discover_topography_files_returns_all_stages(tmp_path):
    module = load_module()
    topo_dir = tmp_path / "products"
    topo_dir.mkdir()
    for label in module.TOPO_SEQUENCE:
        (topo_dir / f"sacramento_valley_topo_{label}.pt").write_text("", encoding="utf-8")

    mapping = module.discover_topography_files(str(topo_dir))

    assert list(mapping) == list(module.TOPO_SEQUENCE)
    assert mapping["2p4km"].name == "sacramento_valley_topo_2p4km.pt"


def test_discover_topography_files_can_select_single_stage(tmp_path):
    module = load_module()
    topo_dir = tmp_path / "products"
    topo_dir.mkdir()
    (topo_dir / "pte1b_topo_2p4km.pt").write_text("", encoding="utf-8")

    mapping = module.discover_topography_files(str(topo_dir), required_labels=("2p4km",))

    assert mapping == {"2p4km": topo_dir / "pte1b_topo_2p4km.pt"}


def test_build_topography_field_flips_north_first_rows():
    module = load_module()
    topo_vars = {
        "topography": torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float64),
        "row0_is_north": torch.tensor([1], dtype=torch.int8),
    }

    padded = module.build_topography_field(topo_vars, (2, 2), nghost=1)

    assert padded.shape == (4, 4, 1)
    interior = padded[1:-1, 1:-1, 0]
    expected = torch.tensor([[30.0, 40.0], [10.0, 20.0]], dtype=torch.float64)
    assert torch.equal(interior, expected)


def test_implicit_scheme_for_refinement_factor():
    module = load_module()

    assert module.implicit_scheme_for_refinement_factor(1.0) == 1
    assert module.implicit_scheme_for_refinement_factor(2.0) == 1
    assert module.implicit_scheme_for_refinement_factor(4.0) == 1
    assert module.implicit_scheme_for_refinement_factor(8.0) == 1


def test_refined_global_horizontal_cells_uses_layout_counts():
    module = load_module()

    assert module.refined_global_horizontal_cells(28, 46, 2, 1) == (112, 92)
    assert module.refined_global_horizontal_cells(56, 92, 2, 1) == (224, 184)
    assert module.refined_global_horizontal_cells(56, 46, 1, 1) == (112, 92)


def test_precipitation_tracer_indices_reads_particle_species(tmp_path):
    module = load_module()
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
species:
  - name: dry
  - name: H2O
  - name: H2O(l)
  - name: H2O(l,p)
  - name: dust(p)
""",
        encoding="utf-8",
    )

    indices = module.precipitation_tracer_indices(str(config_file))

    assert indices == (module.kICY + 2, module.kICY + 3)


def test_remove_surface_precipitation_zeros_lowest_fluid_cells():
    module = load_module()
    hydro_w = torch.zeros((8, 2, 1, 4), dtype=torch.float64)
    hydro_w[module.kICY + 2] = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]], dtype=torch.float64
    )
    topo = torch.tensor(
        [[[False, False, False, False]], [[True, True, False, False]]], dtype=torch.bool
    )

    updated = module.remove_surface_precipitation(
        hydro_w, topo, (module.kICY + 2,)
    )

    expected = torch.tensor(
        [[[0.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 0.0, 8.0]]], dtype=torch.float64
    )
    assert torch.equal(updated[module.kICY + 2], expected)
    assert torch.equal(updated[module.kICY], hydro_w[module.kICY])


def test_clone_with_lateral_ghost_zones_updates_only_boundary_cells():
    module = load_module()
    interior = torch.zeros((2, 6, 6, 3), dtype=torch.float64)
    boundary = torch.ones((2, 6, 6, 3), dtype=torch.float64)

    updated = module.clone_with_lateral_ghost_zones(interior, boundary, nghost=1)

    assert torch.equal(updated[:, 1:-1, 1:-1, :], interior[:, 1:-1, 1:-1, :])
    assert torch.all(updated[:, 0, :, :] == 1.0)
    assert torch.all(updated[:, -1, :, :] == 1.0)
    assert torch.all(updated[:, :, 0, :] == 1.0)
    assert torch.all(updated[:, :, -1, :] == 1.0)


def test_refine_boundary_state_to_match_doubles_horizontal_resolution():
    module = load_module()
    boundary = torch.arange(2 * 8 * 8 * 3, dtype=torch.float64).reshape(2, 8, 8, 3)

    refined = module.refine_boundary_state_to_match(boundary, (2, 12, 12, 3), nghost=2)

    assert refined.shape == (2, 12, 12, 3)
    assert torch.isfinite(refined).all()


def test_refine_boundary_state_to_match_reaches_two_refinement_levels():
    module = load_module()
    boundary = torch.arange(2 * 8 * 8 * 3, dtype=torch.float64).reshape(2, 8, 8, 3)

    refined = module.refine_boundary_state_to_match(boundary, (2, 20, 20, 3), nghost=2)

    assert refined.shape == (2, 20, 20, 3)
    assert torch.isfinite(refined).all()


def test_selected_topography_labels_for_no_refinement():
    module = load_module()

    assert module.selected_topography_labels("none", None) == ("2p4km",)
    assert module.selected_topography_labels("none", "1p2km") == ("1p2km",)
    assert module.selected_topography_labels("staged", None) == module.TOPO_SEQUENCE


def test_run_simulation_resynchronizes_hydro_w_after_state_updates():
    module = load_module()

    class FakeIntegrator:
        stages = (0,)

        def stop(self, cycle, current_time):
            return current_time >= 1.0

    class FakeEOS:
        def compute(self, op, args):
            assert op == "U->W"
            return args[0] + 100.0

    class FakeCoord:
        def nghost(self):
            return 1

    class FakeOptions:
        class _IntgOptions:
            def tlim(self, value):
                self.value = value

        def coord(self):
            return FakeCoord()

        def intg(self):
            return self._IntgOptions()

    class FakeBlock:
        def __init__(self):
            self._cycle = 0
            self._options = FakeOptions()
            self.eos = FakeEOS()

        def module(self, name):
            if name == "hydro.eos":
                return self.eos
            if name == "hydro.eos.thermo":
                return object()
            raise KeyError(name)

        def cycle(self):
            return self._cycle

        def options(self):
            return self._options

        @property
        def options(self):
            return self._options

        def apply_hydro_bc(self, hydro_w, type=0):
            hydro_w[..., 0] += 1.0

    class FakeMesh:
        def __init__(self):
            self.blocks = [FakeBlock()]
            self.intg = FakeIntegrator()
            self.output_snapshots = []

        def module(self, name):
            if name == "block0.intg":
                return self.intg
            raise KeyError(name)

        def set_cycle(self, cycle):
            self.blocks[0]._cycle = cycle

        def max_time_step(self, mesh_vars):
            return 1.0

        def print_cycle_info(self, mesh_vars, current_time, dt):
            pass

        def forward(self, mesh_vars, dt, stage):
            mesh_vars[0]["hydro_u"] = mesh_vars[0]["hydro_u"] + 2.0

        def check_redo(self, mesh_vars):
            return 0

        def make_outputs(self, mesh_vars, current_time):
            self.output_snapshots.append(
                (
                    current_time,
                    mesh_vars[0]["hydro_u"].clone(),
                    mesh_vars[0]["hydro_w"].clone(),
                )
            )

    mesh = FakeMesh()
    block_vars = {
        "hydro_u": torch.zeros((6, 2, 2, 2), dtype=torch.float64),
        "hydro_w": torch.zeros((6, 2, 2, 2), dtype=torch.float64),
        "topo": torch.zeros((2, 2, 2), dtype=torch.bool),
    }

    original_add = module.add_frigate_forcing
    original_evolve = module.evolve_kinetics
    try:
        module.add_frigate_forcing = (
            lambda block_vars, eos, precip_indices, dt, **kwargs: block_vars["hydro_u"].add_(3.0)
        )
        module.evolve_kinetics = (
            lambda hydro_w, eos, thermo_x, thermo_y, kinet, dt: torch.ones_like(hydro_w[module.kICY:])
        )
        mesh_vars, current_time = module.run_simulation(
            mesh, object(), object(), [block_vars], 0.0, 1.0, ()
        )
    finally:
        module.add_frigate_forcing = original_add
        module.evolve_kinetics = original_evolve

    assert current_time == 1.0
    assert len(mesh.output_snapshots) == 1
    _, hydro_u_out, hydro_w_out = mesh.output_snapshots[0]
    expected_hydro_u = torch.full_like(hydro_u_out, 5.0)
    expected_hydro_u[module.kICY:] += 1.0
    assert torch.equal(hydro_u_out, expected_hydro_u)
    expected_hydro_w = expected_hydro_u + 100.0
    expected_hydro_w[..., 0] += 1.0
    assert torch.equal(hydro_w_out, expected_hydro_w)
    assert torch.equal(mesh_vars[0]["hydro_w"], expected_hydro_w)


def test_apply_surface_heating_warms_only_daylit_surface_cells():
    module = load_module()
    block_vars = {
        "hydro_u": torch.zeros((6, 2, 1, 3), dtype=torch.float64),
        "topo": torch.zeros((2, 1, 3), dtype=torch.bool),
        "surface_lon_deg": torch.zeros((2, 1), dtype=torch.float64),
        "surface_lat_deg": torch.zeros((2, 1), dtype=torch.float64),
        "surface_cell_depth_m": torch.tensor([200.0], dtype=torch.float64),
    }
    solar = module.SolarHeatingConfig(
        start_time_utc=datetime(2026, 3, 20, 0, 0, 0, tzinfo=UTC),
        center_longitude_deg=0.0,
        center_latitude_deg=0.0,
        sensible_heat_flux_w_m2=200.0,
    )

    module.apply_surface_heating(block_vars, current_time=12 * 3600.0, dt=10.0, solar_heating=solar)

    heated = block_vars["hydro_u"][module.kIPR, :, :, 0]
    assert torch.all(heated > 0.0)
    assert torch.allclose(heated, heated[0, 0].expand_as(heated))
    assert torch.all(block_vars["hydro_u"][module.kIPR, :, :, 1:] == 0.0)


def test_apply_surface_heating_turns_off_at_night():
    module = load_module()
    block_vars = {
        "hydro_u": torch.zeros((6, 1, 1, 2), dtype=torch.float64),
        "topo": torch.zeros((1, 1, 2), dtype=torch.bool),
        "surface_lon_deg": torch.zeros((1, 1), dtype=torch.float64),
        "surface_lat_deg": torch.zeros((1, 1), dtype=torch.float64),
        "surface_cell_depth_m": torch.tensor([200.0], dtype=torch.float64),
    }
    solar = module.SolarHeatingConfig(
        start_time_utc=datetime(2026, 3, 20, 0, 0, 0, tzinfo=UTC),
        center_longitude_deg=0.0,
        center_latitude_deg=0.0,
        sensible_heat_flux_w_m2=200.0,
    )

    module.apply_surface_heating(block_vars, current_time=0.0, dt=10.0, solar_heating=solar)

    assert torch.all(block_vars["hydro_u"][module.kIPR] == 0.0)


def test_run_staged_ghost_schedule_can_skip_18h_refinement(monkeypatch):
    module = load_module()
    events = []

    class FakeOutput:
        def dt(self, value):
            events.append(("output_dt", value))

    class FakeCoord:
        def nghost(self):
            return 2

    class FakeOptions:
        def outputs(self):
            return [FakeOutput()]

        def coord(self):
            return FakeCoord()

    class FakeBlock:
        def __init__(self):
            self._options = FakeOptions()

        @property
        def options(self):
            return self._options

    class FakeMesh:
        def __init__(self):
            self.blocks = [FakeBlock()]

    def fake_run_simulation(
        mesh, thermo_x, kinet, mesh_vars, current_time, duration, precip_indices, **kwargs
    ):
        events.append(("run", current_time, duration))
        return mesh_vars, current_time + duration

    def fake_refine_simulation(mesh, block_vars, topo_vars, config_file, **kwargs):
        events.append(("refine", topo_vars["label"]))
        assert kwargs == {"solar_heating": None}
        shape = block_vars["hydro_u"].shape
        next_shape = (shape[0], (shape[1] - 4) * 2 + 4, (shape[2] - 4) * 2 + 4, shape[3])
        refined = {
            "hydro_u": torch.zeros(next_shape, dtype=torch.float64),
            "hydro_w": torch.zeros(next_shape, dtype=torch.float64),
        }
        return mesh, "thermo", "kinet", refined

    def fake_apply_ghost(mesh, block_vars, target_hydro_u):
        events.append(("ghost", tuple(target_hydro_u.shape)))
        return {
            **block_vars,
            "hydro_u": target_hydro_u.clone(),
            "hydro_w": torch.zeros_like(target_hydro_u),
        }

    monkeypatch.setattr(module, "run_simulation", fake_run_simulation)
    monkeypatch.setattr(module, "refine_simulation", fake_refine_simulation)
    monkeypatch.setattr(module, "apply_ghost_zone_forcing", fake_apply_ghost)

    mesh = FakeMesh()
    block_vars = {
        "hydro_u": torch.zeros((2, 8, 8, 3), dtype=torch.float64),
        "hydro_w": torch.zeros((2, 8, 8, 3), dtype=torch.float64),
    }
    ecmwf_hydro_u = [torch.ones((2, 8, 8, 3), dtype=torch.float64) * index for index in range(4)]
    stage_topography = [{"label": label} for label in module.TOPO_SEQUENCE]

    _, _, _, result_vars, current_time = module.run_staged_ghost_schedule(
        mesh,
        "thermo",
        "kinet",
        block_vars,
        ecmwf_hydro_u,
        stage_topography,
        0.0,
        "config.yaml",
        21600.0,
        (),
        refine_at_18h=False,
    )

    assert current_time == 64800.0
    assert result_vars["hydro_u"].shape == (2, 20, 20, 3)
    assert events == [
        ("output_dt", 3600.0),
        ("run", 0.0, 21600.0),
        ("refine", "1p2km"),
        ("ghost", (2, 12, 12, 3)),
        ("run", 21600.0, 21600.0),
        ("refine", "0p6km"),
        ("ghost", (2, 20, 20, 3)),
        ("run", 43200.0, 21600.0),
        ("ghost", (2, 20, 20, 3)),
    ]


def test_run_staged_ghost_schedule_refines_at_18h_by_default(monkeypatch):
    module = load_module()
    events = []

    class FakeOutput:
        def dt(self, value):
            events.append(("output_dt", value))

    class FakeCoord:
        def nghost(self):
            return 2

    class FakeOptions:
        def outputs(self):
            return [FakeOutput()]

        def coord(self):
            return FakeCoord()

    class FakeBlock:
        def __init__(self):
            self._options = FakeOptions()

        @property
        def options(self):
            return self._options

    class FakeMesh:
        def __init__(self):
            self.blocks = [FakeBlock()]

    def fake_run_simulation(
        mesh, thermo_x, kinet, mesh_vars, current_time, duration, precip_indices, **kwargs
    ):
        events.append(("run", current_time, duration))
        return mesh_vars, current_time + duration

    def fake_refine_simulation(mesh, block_vars, topo_vars, config_file, **kwargs):
        events.append(("refine", topo_vars["label"], kwargs))
        shape = block_vars["hydro_u"].shape
        next_shape = (shape[0], (shape[1] - 4) * 2 + 4, (shape[2] - 4) * 2 + 4, shape[3])
        refined = {
            "hydro_u": torch.zeros(next_shape, dtype=torch.float64),
            "hydro_w": torch.zeros(next_shape, dtype=torch.float64),
        }
        return mesh, "thermo", "kinet", refined

    def fake_apply_ghost(mesh, block_vars, target_hydro_u):
        events.append(("ghost", tuple(target_hydro_u.shape)))
        return {
            **block_vars,
            "hydro_u": target_hydro_u.clone(),
            "hydro_w": torch.zeros_like(target_hydro_u),
        }

    monkeypatch.setattr(module, "run_simulation", fake_run_simulation)
    monkeypatch.setattr(module, "refine_simulation", fake_refine_simulation)
    monkeypatch.setattr(module, "apply_ghost_zone_forcing", fake_apply_ghost)

    mesh = FakeMesh()
    block_vars = {
        "hydro_u": torch.zeros((2, 8, 8, 3), dtype=torch.float64),
        "hydro_w": torch.zeros((2, 8, 8, 3), dtype=torch.float64),
    }
    ecmwf_hydro_u = [torch.ones((2, 8, 8, 3), dtype=torch.float64) * index for index in range(4)]
    stage_topography = [{"label": label} for label in module.TOPO_SEQUENCE]

    _, _, _, result_vars, current_time = module.run_staged_ghost_schedule(
        mesh,
        "thermo",
        "kinet",
        block_vars,
        ecmwf_hydro_u,
        stage_topography,
        0.0,
        "config.yaml",
        21600.0,
        (),
    )

    assert current_time == 64800.0
    assert result_vars["hydro_u"].shape == (2, 36, 36, 3)
    assert events == [
        ("output_dt", 3600.0),
        ("run", 0.0, 21600.0),
        ("refine", "1p2km", {"solar_heating": None}),
        ("ghost", (2, 12, 12, 3)),
        ("run", 21600.0, 21600.0),
        ("refine", "0p6km", {"solar_heating": None}),
        ("ghost", (2, 20, 20, 3)),
        ("run", 43200.0, 21600.0),
        ("refine", "0p3km", {"solar_heating": None}),
        ("ghost", (2, 36, 36, 3)),
    ]


def test_prediction_ghost_forcing_selects_hourly_post_spinup_states(monkeypatch):
    module = load_module()
    captured = {}
    times = (0.0, 21600.0, 43200.0, 64800.0, 68400.0, 72000.0, 75600.0)
    states = [torch.full((2, 2, 2, 2), index, dtype=torch.float64) for index in range(len(times))]

    def fake_run_simulation(*args, **kwargs):
        captured["schedule"] = kwargs["forcing_schedule"]
        captured["states"] = kwargs["forcing_states"]
        return args[3], args[4] + args[5]

    monkeypatch.setattr(module, "run_simulation", fake_run_simulation)
    ctx = SimpleNamespace(
        mesh=SimpleNamespace(blocks=[SimpleNamespace(cycle=lambda: 7)]),
        thermo_x="thermo",
        kinet="kinet",
        block_vars={"hydro_u": torch.zeros((2, 2, 2, 2))},
        current_time=64800.0,
        precip_indices=(),
        solar_heating=None,
        forcing_times_seconds=times,
        ecmwf_hydro_u=states,
    )

    module.run_prediction_stage_with_ghost_forcing(ctx, 10800.0)

    assert captured["schedule"].times_seconds == (68400.0, 72000.0, 75600.0)
    assert [int(state.flatten()[0]) for state in captured["states"]] == [4, 5, 6]
    assert ctx.current_time == 75600.0


def test_prediction_ghost_forcing_rejects_insufficient_boundary_coverage():
    module = load_module()
    ctx = SimpleNamespace(
        current_time=64800.0,
        forcing_times_seconds=(0.0, 21600.0, 43200.0, 64800.0, 68400.0),
        ecmwf_hydro_u=[torch.zeros(1) for _ in range(5)],
    )

    try:
        module.run_prediction_stage_with_ghost_forcing(ctx, 7200.0)
    except ValueError as exc:
        assert "do not cover" in str(exc)
    else:
        raise AssertionError("Expected insufficient boundary coverage to fail")


def test_staged_prediction_duration_absorbs_spinup_overshoot():
    module = load_module()
    args = SimpleNamespace(
        refinement_mode="staged",
        spinup_chunk_duration=21600.0,
        prediction_duration=216000.0,
        forecast_end_time_seconds=None,
    )
    ctx = SimpleNamespace(
        current_time=64801.068613154726,
        is_runtime_restart=False,
    )

    remaining = module.remaining_prediction_duration(args, ctx)

    assert remaining == 280800.0 - ctx.current_time
    assert ctx.current_time + remaining == 280800.0


def test_absolute_forecast_end_resumes_from_runtime_restart():
    module = load_module()
    args = SimpleNamespace(
        refinement_mode="staged",
        spinup_chunk_duration=21600.0,
        prediction_duration=216000.0,
        forecast_end_time_seconds=280800.0,
    )
    ctx = SimpleNamespace(
        current_time=64801.068613154726,
        is_runtime_restart=True,
    )

    assert module.remaining_prediction_duration(args, ctx) == 215998.93138684527


def test_run_forecast_uses_staged_ghost_path(monkeypatch):
    module = load_module()
    calls = []

    def fake_build(args):
        calls.append(("build", args.refinement_mode, args.forcing_mode))
        return "ctx"

    def fake_hydro(ctx, duration):
        calls.append(("hydro", ctx, duration))

    def fake_spinup(ctx, config_file, chunk_duration):
        raise AssertionError("staged+nudge path should not run in staged ghost mode")

    def fake_spinup_ghost(ctx, config_file, chunk_duration):
        calls.append(("staged-ghost", ctx, config_file, chunk_duration))

    def fake_prediction(ctx, duration):
        calls.append(("predict", ctx, duration))

    def fake_prediction_ghost(*args, **kwargs):
        raise AssertionError("low-resolution ghost path should not run in staged ghost mode")

    def fake_finalize(ctx):
        calls.append(("finalize", ctx))

    monkeypatch.setattr(module, "build_forecast_context", fake_build)
    monkeypatch.setattr(module, "run_hydrostatic_adjustment", fake_hydro)
    monkeypatch.setattr(module, "run_spinup_stage", fake_spinup)
    monkeypatch.setattr(module, "run_spinup_stage_with_ghost_forcing", fake_spinup_ghost)
    monkeypatch.setattr(module, "run_prediction_stage", fake_prediction)
    monkeypatch.setattr(module, "run_prediction_stage_with_ghost_forcing", fake_prediction_ghost)
    monkeypatch.setattr(module, "finalize_forecast", fake_finalize)

    args = SimpleNamespace(
        refinement_mode="staged",
        forcing_mode="ghost",
        hydrostatic_duration=10.0,
        spinup_chunk_duration=21600.0,
        prediction_duration=86400.0,
        forcing_interval_seconds=21600.0,
        config="dummy.yaml",
    )

    module.run_forecast(args)

    assert calls == [
        ("build", "staged", "ghost"),
        ("hydro", "ctx", 10.0),
        ("staged-ghost", "ctx", "dummy.yaml", 21600.0),
        ("predict", "ctx", 86400.0),
        ("finalize", "ctx"),
    ]


def test_run_forecast_runtime_restart_skips_spinup_and_hydrostatic(monkeypatch):
    module = load_module()
    calls = []

    restart_ctx = SimpleNamespace(
        is_runtime_restart=True,
        mesh=object(),
        block_vars={},
        current_time=86400.0,
    )

    def fake_build(args):
        calls.append(("build", args.input_dir))
        return restart_ctx

    def fake_hydro(*args, **kwargs):
        raise AssertionError("runtime restart should not run hydrostatic adjustment")

    def fake_spinup(*args, **kwargs):
        raise AssertionError("runtime restart should not run spinup")

    def fake_spinup_ghost(*args, **kwargs):
        raise AssertionError("runtime restart should not run staged ghost spinup")

    def fake_prediction(ctx, duration):
        calls.append(("predict", ctx, duration))

    def fake_prediction_ghost(*args, **kwargs):
        raise AssertionError("runtime restart should not enter ghost forcing path")

    def fake_finalize(ctx):
        calls.append(("finalize", ctx))

    monkeypatch.setattr(module, "build_forecast_context", fake_build)
    monkeypatch.setattr(module, "run_hydrostatic_adjustment", fake_hydro)
    monkeypatch.setattr(module, "run_spinup_stage", fake_spinup)
    monkeypatch.setattr(module, "run_spinup_stage_with_ghost_forcing", fake_spinup_ghost)
    monkeypatch.setattr(module, "run_prediction_stage", fake_prediction)
    monkeypatch.setattr(module, "run_prediction_stage_with_ghost_forcing", fake_prediction_ghost)
    monkeypatch.setattr(module, "finalize_forecast", fake_finalize)

    args = SimpleNamespace(
        input_dir="pte1b.final.restart",
        refinement_mode="staged",
        forcing_mode="ghost",
        hydrostatic_duration=10.0,
        spinup_chunk_duration=21600.0,
        prediction_duration=43200.0,
        forcing_interval_seconds=21600.0,
        config="dummy.yaml",
    )

    module.run_forecast(args)

    assert calls == [
        ("build", "pte1b.final.restart"),
        ("predict", restart_ctx, 43200.0),
        ("finalize", restart_ctx),
    ]


def test_run_forecast_uses_low_resolution_ghost_path(monkeypatch):
    module = load_module()
    calls = []

    def fake_build(args):
        calls.append(("build", args.refinement_mode, args.forcing_mode))
        return "ctx"

    def fake_hydro(ctx, duration):
        calls.append(("hydro", ctx, duration))

    def fake_spinup(*args, **kwargs):
        raise AssertionError("staged spinup should not run in low-resolution ghost mode")

    def fake_prediction(*args, **kwargs):
        raise AssertionError("plain prediction should not run in low-resolution ghost mode")

    def fake_prediction_ghost(ctx, duration, interval):
        calls.append(("ghost", ctx, duration, interval))

    def fake_finalize(ctx):
        calls.append(("finalize", ctx))

    monkeypatch.setattr(module, "build_forecast_context", fake_build)
    monkeypatch.setattr(module, "run_hydrostatic_adjustment", fake_hydro)
    monkeypatch.setattr(module, "run_spinup_stage", fake_spinup)
    monkeypatch.setattr(module, "run_prediction_stage", fake_prediction)
    monkeypatch.setattr(module, "run_prediction_stage_with_ghost_forcing", fake_prediction_ghost)
    monkeypatch.setattr(module, "finalize_forecast", fake_finalize)

    args = SimpleNamespace(
        refinement_mode="none",
        forcing_mode="ghost",
        hydrostatic_duration=10.0,
        prediction_duration=86400.0,
        forcing_interval_seconds=21600.0,
        config="dummy.yaml",
    )

    module.run_forecast(args)

    assert calls == [
        ("build", "none", "ghost"),
        ("hydro", "ctx", 10.0),
        ("ghost", "ctx", 86400.0, 21600.0),
        ("finalize", "ctx"),
    ]
