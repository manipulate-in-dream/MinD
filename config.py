"""
Central configuration module for MinD project.
Provides unified path management and configuration loading.
"""
import os
from pathlib import Path
import yaml
import json
from typing import Optional, Dict, Any


class PathConfig:
    """Centralized path configuration for MinD project."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize path configuration.
        
        Args:
            config_file: Optional path to custom config file. 
                        Defaults to config.yaml in project root.
        """
        self.project_root = Path(__file__).parent.absolute()
        
        # Load configuration from file or environment
        self.config = self._load_config(config_file)
        
        # Set up base paths from environment variables or defaults
        self.data_root = Path(os.getenv('MIND_DATA_ROOT', 
                                       self.config.get('data_root', 
                                                      self.project_root / 'data')))
        self.model_root = Path(os.getenv('MIND_MODEL_ROOT', 
                                        self.config.get('model_root', 
                                                       self.project_root / 'checkpoints')))
        self.output_root = Path(os.getenv('MIND_OUTPUT_ROOT', 
                                         self.config.get('output_root', 
                                                        self.project_root / 'outputs')))
        
        # Dataset paths
        self.dataset_root = Path(os.getenv('MIND_DATASET_ROOT', 
                                          self.config.get('dataset_root', 
                                                         self.data_root / 'datasets')))
        
        # Video model paths
        self.video_model_root = self.model_root / 'video_models'
        self.action_model_root = self.model_root / 'action_models'
        self.vla_model_root = self.model_root / 'vla_models'
        
        # DynamiCrafter specific paths
        self.dynamicrafter_root = self.project_root / 'DynamiCrafter'
        self.dynamicrafter_ckpt = self.video_model_root / 'dynamicrafter'
        
        # Output paths
        self.predicted_videos_dir = self.output_root / 'predicted_videos'
        self.evaluation_results_dir = self.output_root / 'evaluation_results'
        self.logs_dir = self.output_root / 'logs'
        
        # Create necessary directories
        self._ensure_directories()
    
    def _load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file."""
        config = {}
        
        # Try to load from specified file or default locations
        config_paths = []
        if config_file:
            config_paths.append(Path(config_file))
        
        # Default config file locations
        config_paths.extend([
            self.project_root / 'config.yaml',
            self.project_root / 'config.json',
            self.project_root / '.mind_config.yaml',
        ])
        
        for config_path in config_paths:
            if config_path.exists():
                if config_path.suffix in ['.yaml', '.yml']:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f) or {}
                        break
                elif config_path.suffix == '.json':
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        break
        
        return config
    
    def _ensure_directories(self):
        """Ensure all necessary directories exist."""
        directories = [
            self.data_root,
            self.model_root,
            self.output_root,
            self.dataset_root,
            self.video_model_root,
            self.action_model_root,
            self.vla_model_root,
            self.predicted_videos_dir,
            self.evaluation_results_dir,
            self.logs_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_type: str, model_name: str) -> Path:
        """Get path for a specific model.
        
        Args:
            model_type: Type of model ('video', 'action', 'vla')
            model_name: Name of the model
            
        Returns:
            Path to the model
        """
        if model_type == 'video':
            return self.video_model_root / model_name
        elif model_type == 'action':
            return self.action_model_root / model_name
        elif model_type == 'vla':
            return self.vla_model_root / model_name
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_dataset_path(self, dataset_name: str) -> Path:
        """Get path for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Path to the dataset
        """
        return self.dataset_root / dataset_name
    
    def resolve_path(self, path: str, base: Optional[Path] = None) -> Path:
        """Resolve a path string to an absolute Path object.
        
        Args:
            path: Path string (can be relative or absolute)
            base: Base path for relative paths. Defaults to project root.
            
        Returns:
            Resolved absolute Path object
        """
        path_obj = Path(path)
        
        # If already absolute, return as-is
        if path_obj.is_absolute():
            return path_obj
        
        # Otherwise, resolve relative to base or project root
        base = base or self.project_root
        return (base / path_obj).resolve()
    
    def validate_path(self, path: Path, must_exist: bool = True) -> bool:
        """Validate that a path exists and is accessible.
        
        Args:
            path: Path to validate
            must_exist: Whether the path must exist
            
        Returns:
            True if path is valid
            
        Raises:
            FileNotFoundError: If path doesn't exist and must_exist is True
        """
        if must_exist and not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        return True
    
    def to_dict(self) -> Dict[str, str]:
        """Convert configuration to dictionary for serialization."""
        return {
            'project_root': str(self.project_root),
            'data_root': str(self.data_root),
            'model_root': str(self.model_root),
            'output_root': str(self.output_root),
            'dataset_root': str(self.dataset_root),
            'video_model_root': str(self.video_model_root),
            'action_model_root': str(self.action_model_root),
            'vla_model_root': str(self.vla_model_root),
            'dynamicrafter_root': str(self.dynamicrafter_root),
            'predicted_videos_dir': str(self.predicted_videos_dir),
            'evaluation_results_dir': str(self.evaluation_results_dir),
            'logs_dir': str(self.logs_dir),
        }


# Global instance for easy import
_path_config = None

def get_path_config(config_file: Optional[str] = None) -> PathConfig:
    """Get or create the global PathConfig instance.
    
    Args:
        config_file: Optional path to custom config file
        
    Returns:
        PathConfig instance
    """
    global _path_config
    if _path_config is None:
        _path_config = PathConfig(config_file)
    return _path_config


def reset_path_config():
    """Reset the global PathConfig instance."""
    global _path_config
    _path_config = None