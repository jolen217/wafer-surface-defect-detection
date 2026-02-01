"""
Configuration module for loading and managing project settings.

This module provides functionality to load configuration from YAML files
and manage environment variables for the wafer defect detection project.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Data configuration settings."""
    dataset_path: str
    processed_path: str
    train_split: float
    val_split: float
    test_split: float
    image_size: list
    batch_size: int
    num_workers: int
    pin_memory: bool


@dataclass
class ModelConfig:
    """Model configuration settings."""
    name: str
    pretrained: bool
    num_classes: int
    dropout: float


@dataclass
class TrainingConfig:
    """Training configuration settings."""
    epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: str
    early_stopping_patience: int
    save_best_model: bool


@dataclass
class AugmentationConfig:
    """Data augmentation configuration settings."""
    enabled: bool
    horizontal_flip: float
    vertical_flip: float
    rotation: int
    brightness: float
    contrast: float
    blur: float


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    use_tensorboard: bool
    use_wandb: bool
    log_dir: str
    save_frequency: int
    save_every_epoch: bool = True
    print_detailed_metrics: bool = True


@dataclass
class DeviceConfig:
    """Device configuration settings."""
    use_cuda: bool
    device_id: int


class Config:
    """Main configuration class that loads and manages all settings."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to configuration YAML file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config_dict = yaml.safe_load(file)
            
            # Initialize configuration objects
            self.data = DataConfig(**config_dict['data'])
            self.model = ModelConfig(**config_dict['model'])
            self.training = TrainingConfig(**config_dict['training'])
            self.augmentation = AugmentationConfig(**config_dict['augmentation'])
            self.logging = LoggingConfig(**config_dict['logging'])
            self.device = DeviceConfig(**config_dict['device'])
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
        except KeyError as e:
            raise ValueError(f"Missing required configuration key: {e}")
    
    def get_project_root(self) -> Path:
        """Get the project root directory."""
        return self.config_path.parent.parent
    
    def get_data_path(self) -> Path:
        """Get the data directory path."""
        return self.get_project_root() / self.data.dataset_path
    
    def get_processed_data_path(self) -> Path:
        """Get the processed data directory path."""
        return self.get_project_root() / self.data.processed_path
    
    def get_log_path(self) -> Path:
        """Get the logging directory path."""
        return self.get_project_root() / self.logging.log_dir
    
    def update_num_classes(self, num_classes: int):
        """Update the number of classes in model configuration."""
        self.model.num_classes = num_classes
    
    def save_config(self, path: str = None):
        """Save current configuration to YAML file."""
        if path is None:
            path = self.config_path
        
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'augmentation': self.augmentation.__dict__,
            'logging': self.logging.__dict__,
            'device': self.device.__dict__
        }
        
        with open(path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)


def get_config() -> Config:
    """Factory function to get configuration instance."""
    return Config()