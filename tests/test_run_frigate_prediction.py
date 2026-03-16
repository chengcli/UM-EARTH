from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

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


def test_build_topography_field_flips_north_first_rows():
    module = load_module()
    topo_vars = {
        "topography": torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float64),
        "row0_is_north": torch.tensor([1], dtype=torch.int8),
    }

    padded = module.build_topography_field(topo_vars, (2, 2), nghost=1)

    assert padded.shape == (4, 4, 1)
    interior = padded[1:-1, 1:-1, 0]
    expected = torch.tensor([[30.0, 10.0], [40.0, 20.0]], dtype=torch.float64)
    assert torch.equal(interior, expected)


def test_implicit_scheme_for_refinement_factor():
    module = load_module()

    assert module.implicit_scheme_for_refinement_factor(1.0) == 1
    assert module.implicit_scheme_for_refinement_factor(2.0) == 1
    assert module.implicit_scheme_for_refinement_factor(4.0) == 1
    assert module.implicit_scheme_for_refinement_factor(8.0) == 1


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
