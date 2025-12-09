"""
Configuration management for SPP Metamaterial PINN project.

This module provides centralized configuration handling with YAML-based
parameter files and runtime configuration management.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch


class ConfigManager:
    """
    Centralized configuration manager for PINN training and physics parameters.
    
    Handles loading, merging, and validation of configuration files with
    support for environment-specific overrides and runtime parameter updates.
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            config_dir = Path(__file__).parent
        self.config_dir = Path(config_dir)
        self._config = {}
        self._loaded_files = []
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Name of configuration file (with or without .yaml extension)
            
        Returns:
            Dictionary containing configuration parameters
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        if not config_file.endswith('.yaml'):
            config_file += '.yaml'
        
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self._loaded_files.append(config_file)
            return config or {}
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing {config_file}: {e}")
    
    def load_base_config(self) -> Dict[str, Any]:
        """Load base configuration file."""
        return self.load_config('base_config.yaml')
    
    def load_metamaterial_config(self) -> Dict[str, Any]:
        """Load metamaterial parameters configuration."""
        return self.load_config('metamaterial_params.yaml')
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.
        
        Later configurations override earlier ones for conflicting keys.
        
        Args:
            *configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        
        for config in configs:
            merged = self._deep_merge(merged, config)
        
        return merged
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            dict1: Base dictionary
            dict2: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def load_full_config(self) -> Dict[str, Any]:
        """
        Load and merge all configuration files.
        
        Returns:
            Complete merged configuration
        """
        base_config = self.load_base_config()
        metamaterial_config = self.load_metamaterial_config()
        
        # Merge configurations with metamaterial params taking precedence
        full_config = self.merge_configs(base_config, metamaterial_config)
        
        # Validate and process configuration
        full_config = self._validate_config(full_config)
        
        self._config = full_config
        return full_config
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and process configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Validated and processed configuration
            
        Raises:
            ValueError: If configuration validation fails
        """
        # Convert complex numbers from strings
        config = self._process_complex_numbers(config)
        
        # Convert scientific notation strings to numbers
        config = self._process_numeric_values(config)
        
        # Validate required sections
        required_sections = ['physics', 'training', 'domain', 'metamaterial']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Required configuration section missing: {section}")
        
        # Validate physics parameters
        physics = config['physics']
        if float(physics['frequency']) <= 0:
            raise ValueError("Physics frequency must be positive")
        
        # Validate training parameters
        training = config['training']
        if float(training['epochs']) <= 0:
            raise ValueError("Training epochs must be positive")
        if float(training['learning_rate']) <= 0:
            raise ValueError("Learning rate must be positive")
        
        # Set device
        # Always prioritize CUDA if available, otherwise fallback to CPU
        if torch.cuda.is_available():
            config['device'] = 'cuda'
            actual_device = 'cuda'
        else:
            config['device'] = 'cpu'
            actual_device = 'cpu'

        # Set PyTorch's default device globally
        torch.set_default_device(actual_device)
        
        return config
    
    def _process_numeric_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process numeric strings in configuration (including scientific notation).
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with processed numeric values
        """
        def convert_numeric(obj):
            if isinstance(obj, dict):
                return {k: convert_numeric(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numeric(item) for item in obj]
            elif isinstance(obj, str):
                # Try to convert scientific notation and regular numbers
                try:
                    # Check for complex numbers first
                    if 'j' in obj or 'i' in obj:
                        complex_str = obj.replace('i', 'j')
                        return complex(complex_str)
                    # Check for scientific notation or regular numbers
                    elif 'e' in obj.lower() or obj.replace('.', '').replace('-', '').isdigit():
                        return float(obj)
                except (ValueError, TypeError):
                    pass
                return obj
            else:
                return obj
        
        return convert_numeric(config)
    
    def _process_complex_numbers(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process complex number strings in configuration.
        
        Converts strings like "-2.0+0.1j" to complex numbers.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with processed complex numbers
        """
        def convert_complex(obj):
            if isinstance(obj, dict):
                return {k: convert_complex(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_complex(item) for item in obj]
            elif isinstance(obj, str):
                # Try to convert complex number strings
                try:
                    if 'j' in obj or 'i' in obj:
                        # Replace 'i' with 'j' for Python complex format
                        complex_str = obj.replace('i', 'j')
                        return complex(complex_str)
                except ValueError:
                    pass
                return obj
            else:
                return obj
        
        return convert_complex(config)
    
    def _process_numeric_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process numeric strings in configuration (including scientific notation).
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with processed numeric values
        """
        def convert_numeric(obj):
            if isinstance(obj, dict):
                return {k: convert_numeric(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numeric(item) for item in obj]
            elif isinstance(obj, str):
                # Try to convert scientific notation and regular numbers
                try:
                    # Check for complex numbers first
                    if 'j' in obj or 'i' in obj:
                        complex_str = obj.replace('i', 'j')
                        return complex(complex_str)
                    # Check for scientific notation or regular numbers
                    elif ('e' in obj.lower() or 
                          obj.replace('.', '').replace('-', '').replace('+', '').replace('e', '').replace('E', '').isdigit()):
                        return float(obj)
                except (ValueError, TypeError):
                    pass
                return obj
            else:
                return obj
        
        return convert_numeric(config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Supports dot notation for nested keys (e.g., 'training.lr').
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if not self._config:
            self.load_full_config()
        
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.
        
        Supports dot notation for nested keys.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        if not self._config:
            self.load_full_config()
        
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set final key
        config[keys[-1]] = value
    
    def save_config(self, filename: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            filename: Output filename
            config: Configuration to save (uses current config if None)
        """
        if config is None:
            config = self._config
        
        if not filename.endswith('.yaml'):
            filename += '.yaml'
        
        output_path = self.config_dir / filename
        
        # Convert complex numbers to strings for YAML serialization
        serializable_config = self._make_yaml_serializable(config)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(serializable_config, f, default_flow_style=False, indent=2)
    
    def _make_yaml_serializable(self, obj: Any) -> Any:
        """
        Convert object to YAML-serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            YAML-serializable object
        """
        if isinstance(obj, dict):
            return {k: self._make_yaml_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_yaml_serializable(item) for item in obj]
        elif isinstance(obj, complex):
            return f"{obj.real:+.3f}{obj.imag:+.3f}j"
        else:
            return obj
    
    def print_config(self, section: Optional[str] = None) -> None:
        """
        Print configuration in formatted manner.
        
        Args:
            section: Specific section to print (prints all if None)
        """
        if not self._config:
            self.load_full_config()
        
        config_to_print = self._config if section is None else self._config.get(section, {})
        
        print("=" * 60)
        print(f"Configuration{f' - {section}' if section else ''}")
        print("=" * 60)
        print(yaml.dump(config_to_print, default_flow_style=False, indent=2))
        print("=" * 60)
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get current configuration."""
        if not self._config:
            self.load_full_config()
        return self._config
    
    @property
    def loaded_files(self) -> list:
        """Get list of loaded configuration files."""
        return self._loaded_files.copy()


# Global configuration manager instance
config_manager = ConfigManager()

# Convenience functions for common operations
def load_config() -> Dict[str, Any]:
    """Load full configuration."""
    return config_manager.load_full_config()

def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value by key."""
    return config_manager.get(key, default)

def print_config(section: Optional[str] = None) -> None:
    """Print configuration."""
    config_manager.print_config(section)


__all__ = [
    'ConfigManager', 
    'config_manager', 
    'load_config', 
    'get_config', 
    'print_config'
]