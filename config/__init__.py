"""
Configuration management for SPP Metamaterial PINN project.

This module provides centralized configuration handling with YAML-based
parameter files and runtime configuration management.

Two kinds of configuration file live in this directory:

* ``base_config.yaml`` and ``metamaterial_params.yaml`` follow the nested
  project schema (``physics``, ``training``, ``domain``, ``metamaterial`` ...)
  and are merged and validated by :meth:`ConfigManager.load_full_config`.
* ``spp_config.yaml`` is a standalone, flat experiment config consumed directly
  by ``scripts/train_spp_pinn.py``.  It can be read through
  :meth:`ConfigManager.load_raw`, which performs no schema validation.

Loading configuration has no global side effects: in particular it never
touches ``torch.set_default_device``.  Use :attr:`ConfigManager.device` to
resolve the configured device and apply it yourself.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml

# A plain real number, optionally in scientific notation: "1e-6", "2.2e16",
# "-5", "0.5e15", "+3.0".  PyYAML 1.1 only recognises floats with a decimal
# point *and* a signed exponent, so "2.2e16" and "1.0e14" arrive as strings.
_REAL_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")

# A complex literal such as "-2.0+0.1j", "1e6-0j", "0.5i", "3j".  Only the
# characters a numeric literal may contain are permitted, so ordinary words
# containing an "i" or "j" (e.g. "silica", "impedance") are never touched.
_COMPLEX_RE = re.compile(r"^[-+]?[\d.]+([eE][-+]?\d+)?([-+][\d.]+([eE][-+]?\d+)?)?[ij]$")


class ConfigManager:
    """
    Centralized configuration manager for PINN training and physics parameters.

    Handles loading, merging, and validation of configuration files with
    support for environment-specific overrides and runtime parameter updates.
    """

    REQUIRED_SECTIONS = ("physics", "training", "domain", "metamaterial")

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            config_dir = Path(__file__).parent
        self.config_dir = Path(config_dir)
        self._config: Dict[str, Any] = {}
        self._loaded_files: list = []

    # ------------------------------------------------------------------ loading

    def _resolve_path(self, config_file: str) -> Path:
        if not config_file.endswith((".yaml", ".yml")):
            config_file += ".yaml"
        config_path = self.config_dir / config_file
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        return config_path

    def _read_yaml(self, config_path: Path) -> Dict[str, Any]:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing {config_path.name}: {e}") from e
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"{config_path.name} must contain a YAML mapping at top level")
        return data

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load a single configuration file from the config directory.

        The file's contents are returned as parsed by YAML; no numeric-string
        conversion or schema validation is applied.  The filename is recorded
        in :attr:`loaded_files`.

        Args:
            config_file: Name of configuration file (with or without .yaml extension)

        Returns:
            Dictionary containing configuration parameters

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_path = self._resolve_path(config_file)
        config = self._read_yaml(config_path)
        self._loaded_files.append(config_path.name)
        return config

    def load_raw(self, config_file: str) -> Dict[str, Any]:
        """
        Load an arbitrary named YAML file without validation or processing.

        Intended for standalone experiment configs (e.g. ``spp_config.yaml``)
        whose schema differs from the nested project schema.  Nothing is
        cached on the manager and :attr:`loaded_files` is not updated.

        Args:
            config_file: Name of configuration file (with or without .yaml extension)

        Returns:
            Dictionary exactly as parsed by ``yaml.safe_load``
        """
        return self._read_yaml(self._resolve_path(config_file))

    def load_base_config(self) -> Dict[str, Any]:
        """Load base configuration file."""
        return self.load_config("base_config.yaml")

    def load_metamaterial_config(self) -> Dict[str, Any]:
        """Load metamaterial parameters configuration."""
        return self.load_config("metamaterial_params.yaml")

    def load_full_config(self) -> Dict[str, Any]:
        """
        Load, merge, process and validate all project configuration files.

        Metamaterial parameters take precedence over the base configuration.
        Numeric strings are converted to ``float``/``complex`` and the result
        is validated against the project schema.  The validated config is
        cached and returned.

        Raises:
            ValueError: If configuration validation fails
        """
        base_config = self.load_base_config()
        metamaterial_config = self.load_metamaterial_config()
        full_config = self.merge_configs(base_config, metamaterial_config)
        full_config = self._validate_config(full_config)
        self._config = full_config
        return full_config

    # ------------------------------------------------------------------ merging

    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep-merge configuration dictionaries; later ones override earlier ones.

        Inputs are not mutated.
        """
        merged: Dict[str, Any] = {}
        for config in configs:
            merged = self._deep_merge(merged, config)
        return merged

    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(dict1)
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    # --------------------------------------------------------------- processing

    @staticmethod
    def parse_numeric(value: Any) -> Any:
        """
        Convert a numeric-looking string to ``float`` or ``complex``.

        Non-string values and strings that are not numeric literals are
        returned unchanged.  Examples::

            "1e-6"       -> 1e-06
            "2.2e16"     -> 2.2e16
            "-2.0+0.1j"  -> (-2+0.1j)
            "1.0+0.0i"   -> (1+0j)
            "1.0.0"      -> "1.0.0"   (unchanged)
        """
        if not isinstance(value, str):
            return value
        s = value.strip()
        if _REAL_RE.match(s):
            return float(s)
        if _COMPLEX_RE.match(s):
            try:
                return complex(s.replace("i", "j"))
            except ValueError:
                return value
        return value

    def _process_numeric_values(self, config: Any) -> Any:
        """Recursively apply :meth:`parse_numeric` to every leaf of ``config``."""
        if isinstance(config, dict):
            return {k: self._process_numeric_values(v) for k, v in config.items()}
        if isinstance(config, list):
            return [self._process_numeric_values(item) for item in config]
        return self.parse_numeric(config)

    # --------------------------------------------------------------- validation

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process numeric strings and validate the nested project schema.

        This is a pure function of ``config``: it has no global side effects.

        Raises:
            ValueError: If configuration validation fails
        """
        config = self._process_numeric_values(config)

        for section in self.REQUIRED_SECTIONS:
            if section not in config or not isinstance(config[section], dict):
                raise ValueError(f"Required configuration section missing: {section}")

        physics = config["physics"]
        if "frequency" not in physics:
            raise ValueError("physics.frequency is required")
        if float(physics["frequency"]) <= 0:
            raise ValueError("Physics frequency must be positive")

        training = config["training"]
        for key in ("epochs", "learning_rate"):
            if key not in training:
                raise ValueError(f"training.{key} is required")
        if float(training["epochs"]) <= 0:
            raise ValueError("Training epochs must be positive")
        if float(training["learning_rate"]) <= 0:
            raise ValueError("Learning rate must be positive")

        return config

    # ------------------------------------------------------------------- access

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dotted key (e.g. ``'training.learning_rate'``).

        Returns ``default`` when the key is missing *or* when the key is
        present with a YAML ``null`` value, so ``get(key, x)`` is never
        ``None`` unless ``x`` is.
        """
        if not self._config:
            self.load_full_config()

        value: Any = self._config
        for k in key.split("."):
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return default if value is None else value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dotted key, creating intermediate sections."""
        if not self._config:
            self.load_full_config()

        keys = key.split(".")
        node = self._config
        for k in keys[:-1]:
            if not isinstance(node.get(k), dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value

    @property
    def device(self) -> torch.device:
        """
        Resolve the configured compute device without applying it globally.

        Reads ``hardware.device``; ``"auto"`` (or absent) selects CUDA when
        available, otherwise CPU.  Callers decide what to do with it, e.g.
        ``model.to(cfg.device)`` or ``torch.set_default_device(cfg.device)``.
        """
        requested = self.get("hardware.device", "auto")
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(str(requested))

    @property
    def config(self) -> Dict[str, Any]:
        """Get current (validated) configuration, loading it on first access."""
        if not self._config:
            self.load_full_config()
        return self._config

    @property
    def loaded_files(self) -> list:
        """Names of files loaded via ``load_config``/``load_full_config``."""
        return self._loaded_files.copy()

    # --------------------------------------------------------------- persistence

    def save_config(self, filename: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Save a configuration (current one by default) to YAML in ``config_dir``."""
        if config is None:
            config = self._config
        if not filename.endswith((".yaml", ".yml")):
            filename += ".yaml"
        output_path = self.config_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self._make_yaml_serializable(config), f, default_flow_style=False, indent=2)

    def _make_yaml_serializable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._make_yaml_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_yaml_serializable(item) for item in obj]
        if isinstance(obj, complex):
            return f"{obj.real:+.6g}{obj.imag:+.6g}j"
        return obj

    def print_config(self, section: Optional[str] = None) -> None:
        """Print configuration (or one section) as YAML."""
        cfg = self.config if section is None else self.config.get(section, {})
        print("=" * 60)
        print(f"Configuration{f' - {section}' if section else ''}")
        print("=" * 60)
        print(yaml.dump(self._make_yaml_serializable(cfg), default_flow_style=False, indent=2))
        print("=" * 60)


# Global configuration manager instance
config_manager = ConfigManager()


def load_config() -> Dict[str, Any]:
    """Load full configuration."""
    return config_manager.load_full_config()


def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value by key."""
    return config_manager.get(key, default)


def print_config(section: Optional[str] = None) -> None:
    """Print configuration."""
    config_manager.print_config(section)


__all__ = ["ConfigManager", "config_manager", "load_config", "get_config", "print_config"]
