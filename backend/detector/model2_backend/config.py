#!/usr/bin/env python3
"""
Configuration file for Manual Detection System
Configure your model and input settings here.
"""

import os
from pathlib import Path

class ManualDetectionConfig:
    """Configuration class for manual detection system"""
    
    def __init__(self):
        # Model Configuration
        self.model_path = "trained_manual_detector.pkl"  # Use the .pkl file, not .npz
        self.threshold = 0.7
        self.detection_methods = ['template_matching', 'feature_matching']
        
        # Training Configuration
        self.template_path = "your_manual.jpg"  # Path to your template manual
        self.positive_samples_folder = "training_data/positive"
        self.negative_samples_folder = "training_data/negative"
        
        # Input/Output Configuration
        self.input_path = "test_images"  # Can be file or folder
        self.output_folder = "results"
        self.save_visualizations = True
        self.save_detection_images = True
        
        # Detection Settings
        self.min_confidence = 0.5
        self.max_detections = 10  # Maximum detections per image
        
        # Performance Settings
        self.process_every_nth_frame = 1  # For video processing
        self.resize_images = True
        self.max_image_size = (800, 600)  # (width, height)
        
        # Visualization Settings
        self.show_confidence_scores = True
        self.show_detection_methods = True
        self.bounding_box_thickness = 2
        self.text_scale = 0.5
        
        # Colors for different detection methods
        self.colors = {
            'template_matching': (0, 255, 0),    # Green
            'feature_matching': (255, 0, 0),     # Blue
            'color_histogram': (0, 0, 255)       # Red
        }

# Create a global configuration instance
config = ManualDetectionConfig()

def load_config_from_file(config_file="manual_detection_config.json"):
    """Load configuration from JSON file"""
    import json
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update config with loaded data
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            print(f"Configuration loaded from {config_file}")
            return True
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
    else:
        print(f"Config file {config_file} not found. Using default settings.")
        return False

def save_config_to_file(config_file="manual_detection_config.json"):
    """Save current configuration to JSON file"""
    import json
    
    config_data = {
        'model_path': config.model_path,
        'threshold': config.threshold,
        'detection_methods': config.detection_methods,
        'template_path': config.template_path,
        'positive_samples_folder': config.positive_samples_folder,
        'negative_samples_folder': config.negative_samples_folder,
        'input_path': config.input_path,
        'output_folder': config.output_folder,
        'save_visualizations': config.save_visualizations,
        'save_detection_images': config.save_detection_images,
        'min_confidence': config.min_confidence,
        'max_detections': config.max_detections,
        'process_every_nth_frame': config.process_every_nth_frame,
        'resize_images': config.resize_images,
        'max_image_size': config.max_image_size,
        'show_confidence_scores': config.show_confidence_scores,
        'show_detection_methods': config.show_detection_methods,
        'bounding_box_thickness': config.bounding_box_thickness,
        'text_scale': config.text_scale,
        'colors': config.colors
    }
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"Configuration saved to {config_file}")
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def create_default_config():
    """Create a default configuration file"""
    print("Creating default configuration file...")
    
    # Set default paths
    config.template_path = "your_manual.jpg"
    config.positive_samples_folder = "training_data/positive"
    config.negative_samples_folder = "training_data/negative"
    config.input_path = "test_images"
    config.output_folder = "results"
    
    # Save the default config
    save_config_to_file("manual_detection_config.json")
    
    print("Default configuration created. Please edit the file to set your paths.")

def validate_config():
    """Validate the current configuration"""
    errors = []
    warnings = []
    
    # Check if template exists
    if not os.path.exists(config.template_path):
        errors.append(f"Template file not found: {config.template_path}")
    
    # Check if model exists (if not training)
    if not os.path.exists(config.model_path):
        warnings.append(f"Model file not found: {config.model_path} (will be created during training)")
    
    # Check if input path exists
    if not os.path.exists(config.input_path):
        errors.append(f"Input path not found: {config.input_path}")
    
    # Check if output folder exists, create if not
    if not os.path.exists(config.output_folder):
        try:
            os.makedirs(config.output_folder, exist_ok=True)
            warnings.append(f"Created output folder: {config.output_folder}")
        except Exception as e:
            errors.append(f"Cannot create output folder: {e}")
    
    # Check if training folders exist
    if not os.path.exists(config.positive_samples_folder):
        warnings.append(f"Positive samples folder not found: {config.positive_samples_folder}")
    
    if not os.path.exists(config.negative_samples_folder):
        warnings.append(f"Negative samples folder not found: {config.negative_samples_folder}")
    
    return errors, warnings

def print_config():
    """Print current configuration"""
    print("Manual Detection Configuration:")
    print("=" * 40)
    print(f"Model Path: {config.model_path}")
    print(f"Threshold: {config.threshold}")
    print(f"Detection Methods: {config.detection_methods}")
    print(f"Template Path: {config.template_path}")
    print(f"Input Path: {config.input_path}")
    print(f"Output Folder: {config.output_folder}")
    print(f"Min Confidence: {config.min_confidence}")
    print(f"Max Detections: {config.max_detections}")
    print(f"Save Visualizations: {config.save_visualizations}")
    print(f"Save Detection Images: {config.save_detection_images}")
    print("=" * 40)

# Example configurations for different use cases
EXAMPLE_CONFIGS = {
    "high_accuracy": {
        "threshold": 0.8,
        "detection_methods": ["template_matching", "feature_matching", "color_histogram"],
        "min_confidence": 0.7,
        "max_detections": 5
    },
    
    "fast_processing": {
        "threshold": 0.6,
        "detection_methods": ["template_matching"],
        "min_confidence": 0.5,
        "max_detections": 3,
        "resize_images": True,
        "max_image_size": (640, 480)
    },
    
    "video_processing": {
        "threshold": 0.7,
        "detection_methods": ["template_matching", "feature_matching"],
        "process_every_nth_frame": 3,
        "resize_images": True,
        "max_image_size": (800, 600)
    }
}

def apply_preset_config(preset_name):
    """Apply a preset configuration"""
    if preset_name in EXAMPLE_CONFIGS:
        preset = EXAMPLE_CONFIGS[preset_name]
        for key, value in preset.items():
            if hasattr(config, key):
                setattr(config, key, value)
        print(f"Applied {preset_name} configuration")
        return True
    else:
        print(f"Preset '{preset_name}' not found. Available presets: {list(EXAMPLE_CONFIGS.keys())}")
        return False