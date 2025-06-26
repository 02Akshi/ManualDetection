#!/usr/bin/env python3
"""
Example: Using Manual Detection with Configuration
This script shows how to configure and use the manual detection system.
"""

import os
import sys
from pathlib import Path

# Import the configurable detector
from manual_detector_with_config import ConfigurableManualDetector
from config import config, apply_preset_config, save_config_to_file, print_config

def example_basic_usage():
    """Example 1: Basic usage with default configuration"""
    print("Example 1: Basic Usage")
    print("=" * 30)
    
    # Create detector with default config
    detector = ConfigurableManualDetector()
    
    # Train the model
    print("Training model...")
    success = detector.train_model()
    
    if success:
        print("Model trained successfully!")
        
        # Process a test image
        if os.path.exists("sample_manual.jpg"):
            print("Testing detection...")
            detector.process_single_image("sample_manual.jpg")
    else:
        print("Training failed. Check your configuration.")

def example_high_accuracy():
    """Example 2: High accuracy configuration"""
    print("\nExample 2: High Accuracy Configuration")
    print("=" * 40)
    
    # Apply high accuracy preset
    apply_preset_config("high_accuracy")
    
    # Customize further
    config.template_path = "your_manual.jpg"
    config.input_path = "high_quality_images"
    config.output_folder = "high_accuracy_results"
    
    # Create detector with high accuracy settings
    detector = ConfigurableManualDetector()
    
    print("High accuracy settings applied:")
    print(f"  Threshold: {config.threshold}")
    print(f"  Detection methods: {config.detection_methods}")
    print(f"  Min confidence: {config.min_confidence}")
    
    # Train and process
    detector.train_model()
    detector.process_batch()

def example_fast_processing():
    """Example 3: Fast processing configuration"""
    print("\nExample 3: Fast Processing Configuration")
    print("=" * 40)
    
    # Apply fast processing preset
    apply_preset_config("fast_processing")
    
    # Customize for speed
    config.template_path = "your_manual.jpg"
    config.input_path = "many_images"
    config.output_folder = "fast_results"
    config.save_visualizations = False  # Skip visualizations for speed
    
    # Create detector with fast settings
    detector = ConfigurableManualDetector()
    
    print("Fast processing settings applied:")
    print(f"  Threshold: {config.threshold}")
    print(f"  Detection methods: {config.detection_methods}")
    print(f"  Max image size: {config.max_image_size}")
    print(f"  Save visualizations: {config.save_visualizations}")
    
    # Train and process
    detector.train_model()
    detector.process_batch()

def example_custom_configuration():
    """Example 4: Custom configuration"""
    print("\nExample 4: Custom Configuration")
    print("=" * 35)
    
    # Set custom configuration
    config.model_path = "my_custom_model.pkl"
    config.threshold = 0.75
    config.detection_methods = ["template_matching", "feature_matching"]
    config.template_path = "my_manual.jpg"
    config.positive_samples_folder = "my_positive_samples"
    config.negative_samples_folder = "my_negative_samples"
    config.input_path = "my_test_images"
    config.output_folder = "my_results"
    config.min_confidence = 0.6
    config.max_detections = 5
    config.save_visualizations = True
    config.save_detection_images = True
    
    # Custom colors
    config.colors = {
        'template_matching': (0, 255, 0),    # Green
        'feature_matching': (255, 0, 255),   # Magenta
        'color_histogram': (255, 255, 0)     # Yellow
    }
    
    # Save custom configuration
    save_config_to_file("my_custom_config.json")
    
    # Create detector with custom config
    detector = ConfigurableManualDetector("my_custom_config.json")
    
    print("Custom configuration applied:")
    print_config()
    
    # Train and process
    detector.train_model()
    detector.process_batch()

def example_programmatic_configuration():
    """Example 5: Programmatic configuration"""
    print("\nExample 5: Programmatic Configuration")
    print("=" * 40)
    
    # Create detector
    detector = ConfigurableManualDetector()
    
    # Configure programmatically
    config.template_path = "your_manual.jpg"
    config.threshold = 0.8
    config.detection_methods = ["template_matching"]
    config.input_path = "single_image.jpg"
    config.output_folder = "programmatic_results"
    config.min_confidence = 0.7
    config.max_detections = 3
    
    print("Programmatic configuration:")
    print(f"  Template: {config.template_path}")
    print(f"  Threshold: {config.threshold}")
    print(f"  Methods: {config.detection_methods}")
    
    # Train model
    if detector.train_model():
        # Process single image
        if os.path.exists(config.input_path):
            detector.process_single_image(config.input_path)
        else:
            print(f"Input image not found: {config.input_path}")

def example_batch_processing():
    """Example 6: Batch processing with different configurations"""
    print("\nExample 6: Batch Processing")
    print("=" * 30)
    
    # Configure for batch processing
    config.template_path = "your_manual.jpg"
    config.input_path = "batch_images"
    config.output_folder = "batch_results"
    config.save_visualizations = True
    config.save_detection_images = True
    config.threshold = 0.7
    config.detection_methods = ["template_matching", "feature_matching"]
    
    # Create detector
    detector = ConfigurableManualDetector()
    
    # Train model first
    if detector.train_model():
        # Process all images in batch
        detector.process_batch()
    else:
        print("Training failed. Cannot proceed with batch processing.")

def create_sample_data():
    """Create sample data for examples"""
    print("Creating sample data for examples...")
    
    # Create directories
    os.makedirs("your_manual.jpg", exist_ok=True)  # This will be replaced with actual file
    os.makedirs("training_data/positive", exist_ok=True)
    os.makedirs("training_data/negative", exist_ok=True)
    os.makedirs("test_images", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Create a sample manual if it doesn't exist
    if not os.path.exists("sample_manual.jpg"):
        import cv2
        import numpy as np
        
        manual = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cv2.putText(manual, "SAMPLE MANUAL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(manual, (10, 10), (390, 290), (0, 0, 0), 2)
        cv2.imwrite("sample_manual.jpg", manual)
        print("Created sample_manual.jpg")
    
    print("Sample data created!")

def main():
    """Main function to run examples"""
    print("Manual Detection Configuration Examples")
    print("=" * 50)
    
    # Create sample data
    create_sample_data()
    
    while True:
        print("\nChoose an example to run:")
        print("1. Basic usage")
        print("2. High accuracy configuration")
        print("3. Fast processing configuration")
        print("4. Custom configuration")
        print("5. Programmatic configuration")
        print("6. Batch processing")
        print("7. Show current configuration")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            example_basic_usage()
        elif choice == '2':
            example_high_accuracy()
        elif choice == '3':
            example_fast_processing()
        elif choice == '4':
            example_custom_configuration()
        elif choice == '5':
            example_programmatic_configuration()
        elif choice == '6':
            example_batch_processing()
        elif choice == '7':
            print_config()
        elif choice == '8':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 