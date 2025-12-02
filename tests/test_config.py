"""
Simple configuration test to debug issues.
"""

def test_basic_config():
    """Test basic configuration loading."""
    print("Testing Basic Configuration Loading")
    print("=" * 50)
    
    try:
        from config import ConfigManager
        
        # Test the manager directly
        manager = ConfigManager()
        
        # Test loading individual files
        print("Loading base config...")
        base_config = manager.load_base_config()
        print(f"Base config loaded. Keys: {list(base_config.keys())}")
        
        print("Loading metamaterial config...")
        meta_config = manager.load_metamaterial_config()
        print(f"Metamaterial config loaded. Keys: {list(meta_config.keys())}")
        
        # Test merging
        print("Merging configurations...")
        merged = manager.merge_configs(base_config, meta_config)
        print(f"✓ Configs merged. Keys: {list(merged.keys())}")
        
        # Test processing
        print("Processing complex numbers...")
        processed_complex = manager._process_complex_numbers(merged)
        
        print("Processing numeric values...")
        processed_numeric = manager._process_numeric_values(processed_complex)
        
        # Check specific values
        print("\nChecking specific values:")
        training = processed_numeric.get('training', {})
        print(f"Training section: {training}")
        
        if 'learning_rate' in training:
            lr = training['learning_rate']
            print(f"Learning rate: {lr} ({type(lr)})")
        else:
            print("Learning rate not found in training section")
            
        if 'epochs' in training:
            epochs = training['epochs']
            print(f"Epochs: {epochs} ({type(epochs)})")
        else:
            print("Epochs not found in training section")
            
        physics = processed_numeric.get('physics', {})
        if 'frequency' in physics:
            freq = physics['frequency']
            print(f"Frequency: {freq} ({type(freq)})")
        else:
            print("Frequency not found in physics section")
            
        # Test metamaterial complex numbers
        metamaterial = processed_numeric.get('metamaterial', {})
        if 'permittivity' in metamaterial:
            perm = metamaterial['permittivity']
            if 'parallel' in perm:
                eps_par = perm['parallel']
                print(f"Parallel permittivity: {eps_par} ({type(eps_par)})")
            else:
                print("Parallel permittivity not found")
        else:
            print("Permittivity section not found")
        
        print("\nBasic configuration test completed!")
        assert True
        
    except Exception as e:
        print(f"Basic configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        assert False


def test_full_loading():
    """Test full configuration loading."""
    print("\nTesting Full Configuration Loading")
    print("=" * 50)
    
    try:
        from config import load_config, get_config
        
        print("Loading full configuration...")
        config = load_config()
        print("Full configuration loaded successfully")
        
        # Test parameter access
        lr = get_config('training.learning_rate')
        print(f"Learning rate: {lr}")
        
        freq = get_config('physics.frequency')
        print(f"Frequency: {freq}")
        
        eps_par = get_config('metamaterial.permittivity.parallel')
        print(f"Parallel permittivity: {eps_par}")
        
        print("Full loading test passed!")
        assert True
        
    except Exception as e:
        print(f"Full loading test failed: {e}")
        import traceback
        traceback.print_exc()
        assert False
