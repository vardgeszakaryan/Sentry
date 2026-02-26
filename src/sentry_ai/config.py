import yaml
from pathlib import Path

def load_config(config_path: str | Path) -> dict:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML config file
        
    Returns:
        Dictionary containing configuration
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
        
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config if config else {}
