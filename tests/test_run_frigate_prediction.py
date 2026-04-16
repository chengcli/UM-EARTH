from __future__ import annotations

import importlib.util
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
            lambda block_vars, eos, precip_indices, dt: block_vars["hydro_u"].add_(3.0)
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


def test_run_staged_ghost_schedule_refines_at_06_and_12_only(monkeypatch):
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

    def fake_run_simulation(mesh, thermo_x, kinet, mesh_vars, current_time, duration, precip_indices):
        events.append(("run", current_time, duration))
        return mesh_vars, current_time + duration

    def fake_refine_simulation(mesh, block_vars, topo_vars, config_file, **kwargs):
        events.append(("refine", topo_vars["label"]))
        assert kwargs == {}
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


def test_run_staged_ghost_schedule_can_refine_at_18h(monkeypatch):
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

    def fake_run_simulation(mesh, thermo_x, kinet, mesh_vars, current_time, duration, precip_indices):
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
        refine_at_18h=True,
    )

    assert current_time == 64800.0
    assert result_vars["hydro_u"].shape == (2, 36, 36, 3)
    assert events == [
        ("output_dt", 3600.0),
        ("run", 0.0, 21600.0),
        ("refine", "1p2km", {}),
        ("ghost", (2, 12, 12, 3)),
        ("run", 21600.0, 21600.0),
        ("refine", "0p6km", {}),
        ("ghost", (2, 20, 20, 3)),
        ("run", 43200.0, 21600.0),
        ("refine", "0p3km", {}),
        ("ghost", (2, 36, 36, 3)),
    ]


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
    monkeypatch.setattr(module.dist, "is_available", lambda: False)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: False)

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
    monkeypatch.setattr(module.dist, "is_available", lambda: False)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: False)

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
