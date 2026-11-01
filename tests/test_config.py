"""Tests for the ``config`` package (ConfigManager and helpers)."""

from pathlib import Path

import pytest
import torch
import yaml

from config import ConfigManager, get_config, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def manager() -> ConfigManager:
    return ConfigManager(CONFIG_DIR)


def _minimal_valid() -> dict:
    return {
        "physics": {"frequency": "1e15"},
        "training": {"epochs": 10, "learning_rate": "1e-3"},
        "domain": {"interface_z": 0.0},
        "metamaterial": {"permittivity": {"parallel": "-5.0+0.2j"}},
    }


def _write_pair(tmp_path: Path, base: dict, meta: dict) -> ConfigManager:
    (tmp_path / "base_config.yaml").write_text(yaml.safe_dump(base))
    (tmp_path / "metamaterial_params.yaml").write_text(yaml.safe_dump(meta))
    return ConfigManager(tmp_path)


# ---------------------------------------------------------------------- loading


def test_load_individual_files(manager):
    base = manager.load_base_config()
    meta = manager.load_metamaterial_config()

    assert {"physics", "training", "domain", "network"} <= set(base)
    assert {"metamaterial", "dielectric", "interface"} <= set(meta)
    assert manager.loaded_files == ["base_config.yaml", "metamaterial_params.yaml"]


def test_load_config_accepts_name_without_extension(manager):
    assert manager.load_config("base_config") == manager.load_config("base_config.yaml")


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ConfigManager(tmp_path).load_config("does_not_exist")


def test_load_config_invalid_yaml_raises(tmp_path):
    (tmp_path / "bad.yaml").write_text("a: [unclosed\n")
    with pytest.raises(yaml.YAMLError):
        ConfigManager(tmp_path).load_config("bad")


def test_load_full_config_real_files(manager):
    cfg = manager.load_full_config()

    assert cfg["physics"]["frequency"] == pytest.approx(1e15)
    assert cfg["training"]["learning_rate"] == pytest.approx(1e-3)
    assert cfg["training"]["epochs"] == 10000
    assert cfg["metamaterial"]["permittivity"]["parallel"] == complex(-5.0, 0.2)
    assert cfg["interface"]["normal"] == [0, 0, 1]
    assert cfg["interface"]["boundary_type"] == "continuous"
    # 'derived' block is documentation only and stays null
    assert cfg["derived"]["spp_properties"]["propagation_length"] is None


def test_module_level_helpers():
    cfg = load_config()
    assert cfg["physics"]["frequency"] == pytest.approx(1e15)
    assert get_config("training.learning_rate") == pytest.approx(1e-3)


# ---------------------------------------------------------------------- merging


def test_merge_is_deep_and_later_wins(manager):
    a = {"x": {"p": 1, "q": 2}, "y": [1, 2], "z": 0}
    b = {"x": {"q": 20, "r": 30}, "y": [3]}

    merged = manager.merge_configs(a, b)

    assert merged == {"x": {"p": 1, "q": 20, "r": 30}, "y": [3], "z": 0}


def test_merge_does_not_mutate_inputs(manager):
    a = {"x": {"p": 1}}
    b = {"x": {"q": 2}}
    manager.merge_configs(a, b)
    assert a == {"x": {"p": 1}}
    assert b == {"x": {"q": 2}}


def test_metamaterial_overrides_base(tmp_path):
    base = _minimal_valid()
    meta = {"metamaterial": {"name": "M"}, "training": {"epochs": 7}}
    mgr = _write_pair(tmp_path, base, meta)

    cfg = mgr.load_full_config()

    assert cfg["training"]["epochs"] == 7
    assert cfg["training"]["learning_rate"] == pytest.approx(1e-3)


# ------------------------------------------------------------------- dotted get


def test_get_dotted_key(manager):
    assert manager.get("training.optimizer.type") == "adam"
    assert manager.get("metamaterial.permittivity.parallel") == complex(-5.0, 0.2)


def test_get_missing_key_returns_default(manager):
    sentinel = object()
    assert manager.get("training.nope", sentinel) is sentinel
    assert manager.get("no.such.section", 5) == 5
    assert manager.get("training.optimizer.type.too_deep", "d") == "d"


def test_get_null_value_returns_default(manager):
    # base_config.yaml has logging.wandb.entity: null
    assert manager.config["logging"]["wandb"]["entity"] is None
    assert manager.get("logging.wandb.entity", "fallback") == "fallback"
    assert manager.get("logging.wandb.entity") is None


def test_get_null_value_from_tmp_config(tmp_path):
    base = _minimal_valid()
    base["logging"] = {"entity": None}
    mgr = _write_pair(tmp_path, base, {})

    assert mgr.get("logging.entity", "x") == "x"


def test_set_creates_nested_path(manager):
    manager.set("a.b.c", 3)
    assert manager.get("a.b.c") == 3


# --------------------------------------------------------- numeric string parsing


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1e-6", 1e-6),
        ("2.2e16", 2.2e16),
        ("1.0e14", 1.0e14),
        ("0.5e15", 0.5e15),
        ("-20.0", -20.0),
        ("+3", 3.0),
        ("2.0+0.1j", complex(2.0, 0.1)),
        ("-5.0+0.2j", complex(-5.0, 0.2)),
        ("1.0+0.0i", complex(1.0, 0.0)),
        ("-1e6+0j", complex(-1e6, 0.0)),
        ("0.5j", 0.5j),
    ],
)
def test_parse_numeric_converts(raw, expected):
    assert ConfigManager.parse_numeric(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["1.0.0", "2024-01-15", "silica", "impedance", "tanh", "auto", "Re(ε_metal) < 0", "", "cuda:0"],
)
def test_parse_numeric_leaves_non_numeric_strings(raw):
    assert ConfigManager.parse_numeric(raw) == raw


def test_parse_numeric_passes_through_non_strings():
    assert ConfigManager.parse_numeric(3) == 3
    assert ConfigManager.parse_numeric(None) is None
    assert ConfigManager.parse_numeric([1, "2"]) == [1, "2"]


def test_process_numeric_values_recurses(manager):
    out = manager._process_numeric_values({"a": ["1e3", {"b": "2+1j", "c": "text"}], "d": 4})
    assert out == {"a": [1000.0, {"b": complex(2, 1), "c": "text"}], "d": 4}


def test_real_yaml_scientific_strings_become_floats(manager):
    cfg = manager.load_full_config()
    # PyYAML 1.1 leaves "2.2e16" (no dot-exponent-sign form) as a string
    assert isinstance(cfg["metamaterial"]["properties"]["plasma_frequency"], float)
    assert cfg["metamaterial"]["properties"]["plasma_frequency"] == pytest.approx(2.2e16)
    assert cfg["validation_materials"]["pec"]["permittivity"] == complex(-1e6, 0)
    # Version strings are not numbers
    assert cfg["metadata"]["version"] == "1.0.0"


# ------------------------------------------------------------------- validation


@pytest.mark.parametrize("missing", ["physics", "training", "domain", "metamaterial"])
def test_missing_required_section_raises(tmp_path, missing):
    base = _minimal_valid()
    del base[missing]
    mgr = _write_pair(tmp_path, base, {})
    with pytest.raises(ValueError, match=missing):
        mgr.load_full_config()


def test_non_positive_frequency_raises(tmp_path):
    base = _minimal_valid()
    base["physics"]["frequency"] = 0
    with pytest.raises(ValueError, match="frequency"):
        _write_pair(tmp_path, base, {}).load_full_config()


def test_non_positive_epochs_raises(tmp_path):
    base = _minimal_valid()
    base["training"]["epochs"] = -1
    with pytest.raises(ValueError, match="epochs"):
        _write_pair(tmp_path, base, {}).load_full_config()


def test_non_positive_learning_rate_raises(tmp_path):
    base = _minimal_valid()
    base["training"]["learning_rate"] = "0.0"
    with pytest.raises(ValueError, match="Learning rate"):
        _write_pair(tmp_path, base, {}).load_full_config()


def test_missing_frequency_key_raises(tmp_path):
    base = _minimal_valid()
    del base["physics"]["frequency"]
    with pytest.raises(ValueError, match="physics.frequency"):
        _write_pair(tmp_path, base, {}).load_full_config()


def test_validation_failure_does_not_cache(tmp_path):
    base = _minimal_valid()
    del base["domain"]
    mgr = _write_pair(tmp_path, base, {})
    with pytest.raises(ValueError):
        mgr.load_full_config()
    assert mgr._config == {}


# --------------------------------------------------------------------- load_raw


def test_load_raw_spp_config_is_flat_and_unprocessed(manager):
    raw = manager.load_raw("spp_config")

    # PyYAML 1.1 leaves "2.9758e15" as a string; load_raw does not convert it
    # (train_spp_pinn.py wraps it in float() itself). The value is the ANGULAR
    # frequency for lambda0 = 633 nm.
    assert isinstance(raw["frequency"], str)
    assert float(raw["frequency"]) == pytest.approx(2.9758e15)
    assert raw["metal_permittivity"] == [-19, 0.53]
    assert raw["x_range"] == [-1.0e-6, 1.0e-6]
    assert raw["training"]["num_epochs"] == 10000
    assert raw["model"]["name"] == "SPPNetwork"
    # not part of the nested schema
    assert "physics" not in raw
    # does not touch cached config or loaded_files
    assert manager.loaded_files == []
    assert manager._config == {}


def test_load_raw_skips_validation(tmp_path):
    (tmp_path / "anything.yaml").write_text("foo: '1e-3'\nbar: null\n")
    raw = ConfigManager(tmp_path).load_raw("anything")
    assert raw == {"foo": "1e-3", "bar": None}


def test_load_raw_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ConfigManager(tmp_path).load_raw("nope")


# ----------------------------------------------------------------------- device


def test_loading_config_does_not_change_torch_default_device(manager):
    before = torch.get_default_device()
    manager.load_full_config()
    _ = manager.device
    load_config()
    after = torch.get_default_device()
    assert after == before
    assert torch.empty(1).device == before


def test_device_property_auto(manager):
    dev = manager.device
    assert isinstance(dev, torch.device)
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev.type == expected
    assert "device" not in manager.config  # no injected top-level key


def test_device_property_explicit(tmp_path):
    base = _minimal_valid()
    base["hardware"] = {"device": "cpu"}
    mgr = _write_pair(tmp_path, base, {})
    assert mgr.device == torch.device("cpu")
