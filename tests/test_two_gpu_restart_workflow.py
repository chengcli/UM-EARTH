from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tarfile

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT_ROOT / "run_frigate_prediction.py"
CONVERT_SCRIPT = PROJECT_ROOT / "um_earth" / "ecmwf_api" / "convert_netcdf_to_tensor.py"


def load_run_module():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location("run_frigate_prediction", RUN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_convert_module():
    spec = importlib.util.spec_from_file_location("convert_netcdf_to_tensor", CONVERT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def save_part(path: Path, hydro_w: torch.Tensor) -> None:
    class TensorModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("hydro_w", hydro_w)

    torch.jit.script(TensorModule()).save(str(path))


def test_convert_directory_bundles_two_rank_restart(tmp_path, monkeypatch):
    module = load_convert_module()
    blocks_dir = tmp_path / "blocks"
    blocks_dir.mkdir()
    for name in ("regridded_case_block_0_0.nc", "regridded_case_block_1_0.nc"):
        (blocks_dir / name).write_text("", encoding="utf-8")

    written = []

    def fake_convert(input_file: str, output_file: str | None = None) -> str:
        output_path = Path(output_file)
        output_path.write_text("part", encoding="utf-8")
        written.append(output_path.name)
        return str(output_path)

    monkeypatch.setattr(module, "convert_netcdf_to_tensor", fake_convert)

    restart_file = tmp_path / "bundle.restart"
    outputs = module.convert_directory(
        str(blocks_dir),
        str(tmp_path / "tensors"),
        n_blocks_x2=2,
        n_blocks_x3=1,
        bundle_restart=str(restart_file),
    )

    assert [Path(path).name for path in outputs] == [
        "regridded_case.block0.00000.part",
        "regridded_case.block1.00000.part",
    ]
    assert written == [
        "regridded_case.block0.00000.part",
        "regridded_case.block1.00000.part",
    ]
    with tarfile.open(restart_file, "r") as archive:
        assert sorted(archive.getnames()) == [
            "regridded_case.block0.00000.part",
            "regridded_case.block1.00000.part",
        ]


def test_load_restart_slice_selects_rank_from_restart_bundle(tmp_path, monkeypatch):
    module = load_run_module()
    rank0 = tmp_path / "regridded_case.block0.00000.part"
    rank1 = tmp_path / "regridded_case.block1.00000.part"
    save_part(rank0, torch.zeros((4, 8, 2, 2, 2), dtype=torch.float64))
    save_part(rank1, torch.ones((4, 8, 2, 2, 2), dtype=torch.float64) * 7.0)

    restart_file = tmp_path / "regridded_case.restart"
    with tarfile.open(restart_file, "w") as archive:
        archive.add(rank0, arcname=rank0.name)
        archive.add(rank1, arcname=rank1.name)

    monkeypatch.setenv("RANK", "1")
    block_vars = module.load_restart_slice(restart_file, index=2)

    assert torch.equal(block_vars["hydro_w"], torch.ones((8, 2, 2, 2), dtype=torch.float64) * 7.0)


def test_build_topography_field_for_block_uses_block_position():
    module = load_run_module()

    class FakeLayoutOptions:
        def __init__(self, rank: int):
            self._rank = rank

        def rank(self):
            return self._rank

        def px(self):
            return 2

        def py(self):
            return 1

    class FakeCoordOptions:
        def __init__(self, x2min: float, x2max: float, x3min: float, x3max: float):
            self._x2min = x2min
            self._x2max = x2max
            self._x3min = x3min
            self._x3max = x3max

        def x2min(self):
            return self._x2min

        def x2max(self):
            return self._x2max

        def x3min(self):
            return self._x3min

        def x3max(self):
            return self._x3max

    class FakeOptions:
        def __init__(self, rank: int, x2min: float, x2max: float, x3min: float, x3max: float):
            self._layout = FakeLayoutOptions(rank)
            self._coord = FakeCoordOptions(x2min, x2max, x3min, x3max)

        def layout(self):
            return self._layout

        def coord(self):
            return self._coord

    class FakeCoord:
        def __init__(self, x2v: torch.Tensor, x3v: torch.Tensor):
            self._buffers = {"x2v": x2v, "x3v": x3v}

        def buffer(self, name: str) -> torch.Tensor:
            return self._buffers[name]

    class FakeLayout:
        def loc_of(self, rank: int):
            return (rank, 0, 0)

    class FakeBlock:
        def __init__(self, rank: int, x2min: float, x2max: float):
            self._coord = FakeCoord(
                torch.tensor([x2min + 0.5, x2max - 0.5], dtype=torch.float64),
                torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float64),
            )
            self._options = FakeOptions(rank, x2min, x2max, 0.0, 4.0)
            self._layout = FakeLayout()

        @property
        def options(self):
            return self._options

        def module(self, name: str):
            if name != "coord":
                raise KeyError(name)
            return self._coord

        def get_layout(self):
            return self._layout

    topo_vars = {
        "topography": torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
            ],
            dtype=torch.float64,
        ),
        "row0_is_north": torch.tensor([0], dtype=torch.int8),
    }

    block0_field = module.build_topography_field_for_block(FakeBlock(0, 0.0, 2.0), topo_vars)
    block1_field = module.build_topography_field_for_block(FakeBlock(1, 2.0, 4.0), topo_vars)

    assert block0_field.shape == (4, 2, 1)
    assert block1_field.shape == (4, 2, 1)
    assert block1_field.mean().item() > block0_field.mean().item()
