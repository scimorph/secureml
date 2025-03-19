"""
Regulation-specific presets for SecureML compliance checks.

This module provides functionality to load and access regulation-specific
presets for various privacy regulations.
"""

import os
import yaml
from typing import Dict, Any, Optional, List

# Directory where preset files are stored
PRESET_DIR = os.path.dirname(os.path.abspath(__file__))

# Cache for loaded presets
_preset_cache: Dict[str, Dict[str, Any]] = {}


def list_available_presets() -> List[str]:
    """
    List all available regulation presets.
    
    Returns:
        List of available preset names (without file extension)
    """
    preset_files = [
        f.split('.')[0] for f in os.listdir(PRESET_DIR)
        if f.endswith('.yaml') and not f.startswith('_')
    ]
    return sorted(preset_files)


def load_preset(regulation: str) -> Dict[str, Any]:
    """
    Load a regulation-specific preset.
    
    Args:
        regulation: Name of the regulation (e.g., 'gdpr', 'ccpa', 'hipaa')
    
    Returns:
        Dictionary containing the regulation preset
        
    Raises:
        FileNotFoundError: If the requested preset doesn't exist
        yaml.YAMLError: If the preset file contains invalid YAML
    """
    regulation = regulation.lower()
    
    # Check cache first
    if regulation in _preset_cache:
        return _preset_cache[regulation]
    
    # Load from file
    preset_path = os.path.join(PRESET_DIR, f"{regulation}.yaml")
    
    try:
        with open(preset_path, 'r') as f:
            preset_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Preset for {regulation.upper()} not found. "
            f"Available presets: {', '.join(list_available_presets())}"
        )
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing {regulation.upper()} preset: {str(e)}")
    
    # Cache the loaded preset
    _preset_cache[regulation] = preset_data
    
    return preset_data


def get_preset_field(regulation: str, field_path: str) -> Any:
    """
    Get a specific field from a regulation preset using dot notation.
    
    Args:
        regulation: Name of the regulation (e.g., 'gdpr', 'ccpa', 'hipaa')
        field_path: Path to the field using dot notation 
                   (e.g., 'personal_data_identifiers' or 'security.encryption_required')
    
    Returns:
        Value of the requested field or None if not found
        
    Example:
        >>> get_preset_field('gdpr', 'personal_data_identifiers')
        ['name', 'email', 'phone', ...]
        >>> get_preset_field('gdpr', 'security.encryption_required')
        True
    """
    preset = load_preset(regulation)
    
    # Handle nested fields with dot notation
    if '.' in field_path:
        parts = field_path.split('.')
        current = preset
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
                
        return current
    else:
        return preset.get(field_path) 