#!/usr/bin/env python3
"""
Enhanced Manual Detection Example with Multi-Angle Support
Demonstrates how to train and use the manual detection system for non-top-view images.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from manual_detector import ManualDetector
from config import config, print_config, validate_config

def create_multi_angle_training_guide():
    """Print guide for creating multi-angle training data"""
    print("\n" + "="*60)
    print("MULTI-ANGLE TRAINING GUIDE")
    print("="*60)
    print("To improve detection of non-top-view images, collect training data from different angles:")
    print()
    print("1. TEMPLATE IMAGES (from different viewing angles):")
    print("   • Top view (0° elevation) - your main template")
    print("   • 30° elevation - manual viewed from above at an angle")
    print("   • 45° elevation - manual viewed from side")
    print("   • 60° elevation - manual viewed from below")
    print("   • Different azimuth angles (0°, 90°, 180°, 270°)")
    print()
    print("2. POSITIVE SAMPLES:")
    print("   • Include manuals from various angles in different backgrounds")
    print("   • Include partially occluded manuals")
    print("   • Include manuals with different lighting conditions")
    print()
    print("3. NEGATIVE SAMPLES:")
    print("   • Similar objects that are not manuals")
    print("   • Empty scenes")
    print("   • Other documents or books")
    print()
    print("4. ANGLE INFORMATION:")
    print("   • Elevation: 0° (top view) to 90° (side view)")
    print("   • Azimuth: 0° to 360° (rotation around vertical axis)")
    print("   • Estimate angles if exact measurements aren't available")
    print("="*60)

def train_multi_angle_model():
    """Train a model with multi-angle support"""
    print("\n=== Training Multi-Angle Model ===")
    
    # Initialize detector
    detector = ManualDetector(threshold=config.threshold)
    
    # Check if template exists
    if not os.path.exists(config.template_path):
        print(f"Template not found: {config.template_path}")
        print("Please update config.py with your template path or create a template image.")
        return None
    
    # Add main template (top view)
    print(f"Adding main template: {config.template_path}")
    detector.add_multi_angle_template(
        config.template_path, 
        {'elevation': 0, 'azimuth': 0}  # Top view
    )
    
    # Add additional angle templates if available
    additional_templates = [
        ("template_30deg.jpg", {'elevation': 30, 'azimuth': 0}),
        ("template_45deg.jpg", {'elevation': 45, 'azimuth': 0}),
        ("template_60deg.jpg", {'elevation': 60, 'azimuth': 0}),
        ("template_side.jpg", {'elevation': 90, 'azimuth': 0}),
    ]
    
    for template_file, angle_info in additional_templates:
        if os.path.exists(template_file):
            print(f"Adding angle template: {template_file}")
            detector.add_multi_angle_template(template_file, angle_info)
    
    # Add positive samples
    if os.path.exists(config.positive_samples_folder):
        positive_samples = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            positive_samples.extend(Path(config.positive_samples_folder).glob(ext))
        
        print(f"Adding {len(positive_samples)} positive samples...")
        for sample_path in positive_samples:
            # For this example, we'll use default angle info
            # In practice, you should provide specific angle info for each sample
            detector.add_training_sample(str(sample_path), is_positive=True, angle_info={'elevation': 0, 'azimuth': 0})
    else:
        print(f"Positive samples folder not found: {config.positive_samples_folder}")
    
    # Add negative samples
    if os.path.exists(config.negative_samples_folder):
        negative_samples = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            negative_samples.extend(Path(config.negative_samples_folder).glob(ext))
        
        print(f"Adding {len(negative_samples)} negative samples...")
        for sample_path in negative_samples:
            detector.add_training_sample(str(sample_path), is_positive=False)
    else:
        print(f"Negative samples folder not found: {config.negative_samples_folder}")
    
    # Save the model
    detector.save_model(config.model_path)
    print(f"Model saved to: {config.model_path}")
    
    return detector

def test_multi_angle_detection(detector, test_image_path):
    """Test detection on an image with detailed results"""
    print(f"\n=== Testing Multi-Angle Detection ===")
    print(f"Test image: {test_image_path}")
    
    try:
        # Use enhanced detection methods
        results, image = detector.detect_manual(
            test_image_path, 
            methods=config.detection_methods
        )
        
        # Analyze results
        total_detections = 0
        best_detection = None
        best_confidence = 0
        
        print("\nDetection Results:")
        print("-" * 50)
        
        for method, matches in results.items():
            print(f"\n{method.upper()}:")
            for i, match in enumerate(matches):
                confidence = match['confidence']
                bbox = match['bbox']
                center = match['center']
                
                print(f"  Detection {i+1}:")
                print(f"    Confidence: {confidence:.3f}")
                print(f"    Bounding Box: {bbox}")
                print(f"    Center: {center}")
                
                # Print angle information
                if 'angle_info' in match and match['angle_info']:
                    angle_info = match['angle_info']
                    print(f"    Template Angle: Elevation={angle_info.get('elevation', 'N/A')}°, Azimuth={angle_info.get('azimuth', 'N/A')}°")
                
                if 'estimated_angle' in match:
                    est_angle = match['estimated_angle']
                    print(f"    Estimated Angle: Elevation={est_angle.get('elevation_estimate', 'N/A')}°")
                
                if 'scale' in match:
                    print(f"    Scale: {match['scale']:.2f}")
                
                if 'matches_count' in match:
                    print(f"    Feature Matches: {match['matches_count']}")
                
                total_detections += 1
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_detection = match
        
        # Summary
        print(f"\nSummary:")
        print(f"Total detections: {total_detections}")
        if best_detection:
            print(f"Best detection confidence: {best_confidence:.3f}")
            print(f"Best detection method: {best_detection['method']}")
        
        # Save visualization
        if config.save_detection_images:
            output_path = f"result_{os.path.basename(test_image_path)}"
            detector.draw_matches(image, results, output_path)
            print(f"Result image saved to: {output_path}")
        
        if config.save_visualizations:
            viz_path = f"visualization_{os.path.basename(test_image_path)}.png"
            detector.save_visualization(image, results, viz_path)
            print(f"Visualization saved to: {viz_path}")
        
        return results
        
    except Exception as e:
        print(f"Error during detection: {e}")
        return None

def demonstrate_angle_invariance():
    """Demonstrate the system's ability to handle different viewing angles"""
    print("\n=== Angle Invariance Demonstration ===")
    
    # Load trained model
    if not os.path.exists(config.model_path):
        print(f"Trained model not found: {config.model_path}")
        print("Please train a model first using train_multi_angle_model()")
        return
    
    detector = ManualDetector(config.model_path)
    
    # Test images from different angles (if available)
    test_images = [
        "test_top_view.jpg",
        "test_30deg.jpg", 
        "test_45deg.jpg",
        "test_60deg.jpg",
        "test_side_view.jpg"
    ]
    
    print("Testing detection on images from different viewing angles:")
    print("-" * 60)
    
    for test_image in test_images:
        if os.path.exists(test_image):
            print(f"\nTesting: {test_image}")
            results = test_multi_angle_detection(detector, test_image)
            
            if results:
                total_matches = sum(len(matches) for matches in results.values())
                if total_matches > 0:
                    print(f"✓ Manual detected successfully")
                else:
                    print(f"✗ No manual detected")
        else:
            print(f"\nSkipping: {test_image} (not found)")
    
    print("\n" + "="*60)
    print("ANGLE INVARIANCE SUMMARY")
    print("="*60)
    print("The enhanced system should detect manuals from various viewing angles.")
    print("If detection fails for certain angles, consider:")
    print("1. Adding more training samples from those angles")
    print("2. Adjusting the detection threshold")
    print("3. Using different detection methods")
    print("4. Improving image quality and lighting")

def train_without_angle_info():
    """Demonstrate training without providing angle information"""
    print("\n=== Training Without Angle Information ===")
    print("This example shows how to train the system when you don't know the viewing angles.")
    
    # Initialize detector
    detector = ManualDetector(threshold=0.7)
    
    # Check if template exists
    if not os.path.exists(config.template_path):
        print(f"Template not found: {config.template_path}")
        print("Please update config.py with your template path or create a template image.")
        return None
    
    # Add template with automatic angle estimation
    print(f"Adding template with automatic angle estimation: {config.template_path}")
    detector.add_multi_angle_template_auto(config.template_path)
    
    # Add positive samples with automatic angle estimation
    if os.path.exists(config.positive_samples_folder):
        positive_samples = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            positive_samples.extend(Path(config.positive_samples_folder).glob(ext))
        
        print(f"Adding {len(positive_samples)} positive samples with automatic angle estimation...")
        for sample_path in positive_samples:
            detector.add_training_sample_auto_angle(str(sample_path), is_positive=True)
    else:
        print(f"Positive samples folder not found: {config.positive_samples_folder}")
    
    # Add negative samples
    if os.path.exists(config.negative_samples_folder):
        negative_samples = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            negative_samples.extend(Path(config.negative_samples_folder).glob(ext))
        
        print(f"Adding {len(negative_samples)} negative samples...")
        for sample_path in negative_samples:
            detector.add_training_sample(str(sample_path), is_positive=False)
    else:
        print(f"Negative samples folder not found: {config.negative_samples_folder}")
    
    # Analyze the training data
    print("\nAnalyzing training data and clustering by estimated angles...")
    clusters = detector.cluster_training_samples_by_angle()
    
    if clusters:
        print("\nTraining data distribution by estimated angles:")
        for cluster_name, samples in clusters.items():
            print(f"  {cluster_name}: {len(samples)} samples")
            # Show a few examples from each cluster
            for i, sample in enumerate(samples[:3]):  # Show first 3 examples
                print(f"    - {os.path.basename(sample['image_path'])} "
                      f"(elevation: {sample['elevation']}°)")
            if len(samples) > 3:
                print(f"    ... and {len(samples) - 3} more")
    
    # Save the model
    model_path = "auto_angle_model.pkl"
    detector.save_model(model_path)
    print(f"\nModel saved to: {model_path}")
    
    return detector

def demonstrate_auto_angle_detection():
    """Demonstrate detection with automatically estimated angles"""
    print("\n=== Automatic Angle Detection Demonstration ===")
    
    # Load trained model
    model_path = "auto_angle_model.pkl"
    if not os.path.exists(model_path):
        print(f"Trained model not found: {model_path}")
        print("Please train a model first using train_without_angle_info()")
        return
    
    detector = ManualDetector(model_path)
    
    # Test images (if available)
    test_images = [
        "test_image.jpg",
        "test_side_view.jpg",
        "test_angled.jpg"
    ]
    
    print("Testing detection on images with automatic angle estimation:")
    print("-" * 60)
    
    for test_image in test_images:
        if os.path.exists(test_image):
            print(f"\nTesting: {test_image}")
            
            # Estimate angle for the test image
            estimated_angle = detector.estimate_angle_from_image(test_image)
            print(f"Auto-estimated angle: Elevation={estimated_angle['elevation']}°, "
                  f"Azimuth={estimated_angle['azimuth']}° "
                  f"(confidence: {estimated_angle.get('confidence', 0):.2f})")
            
            # Perform detection
            results = test_multi_angle_detection(detector, test_image)
            
            if results:
                total_matches = sum(len(matches) for matches in results.values())
                if total_matches > 0:
                    print(f"✓ Manual detected successfully")
                else:
                    print(f"✗ No manual detected")
        else:
            print(f"\nSkipping: {test_image} (not found)")
    
    print("\n" + "="*60)
    print("AUTOMATIC ANGLE ESTIMATION SUMMARY")
    print("="*60)
    print("The system automatically estimates viewing angles using:")
    print("1. Aspect ratio analysis (compression indicates angle)")
    print("2. Edge distribution analysis")
    print("3. Perspective line detection")
    print("4. Contour shape analysis")
    print("\nThis allows training without manual angle annotation!")

def main():
    """Main example function"""
    print("Enhanced Manual Detection System - Multi-Angle Example")
    print("=" * 60)
    
    # Print configuration
    print_config()
    
    # Validate configuration
    errors, warnings = validate_config()
    if errors:
        print("\nConfiguration errors:")
        for error in errors:
            print(f"  ✗ {error}")
        print("\nPlease fix these errors before proceeding.")
        return
    
    if warnings:
        print("\nConfiguration warnings:")
        for warning in warnings:
            print(f"  ⚠ {warning}")
    
    # Show training guide
    create_multi_angle_training_guide()
    
    # Ask user what to do
    print("\nChoose an option:")
    print("1. Train multi-angle model")
    print("2. Test detection on single image")
    print("3. Demonstrate angle invariance")
    print("4. Train without angle information")
    print("5. Demonstrate automatic angle detection")
    print("6. Run complete example")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == '1':
        train_multi_angle_model()
    elif choice == '2':
        test_image = input("Enter path to test image: ").strip()
        if os.path.exists(test_image):
            detector = ManualDetector(config.model_path) if os.path.exists(config.model_path) else train_multi_angle_model()
            if detector:
                test_multi_angle_detection(detector, test_image)
        else:
            print(f"Test image not found: {test_image}")
    elif choice == '3':
        demonstrate_angle_invariance()
    elif choice == '4':
        train_without_angle_info()
    elif choice == '5':
        demonstrate_auto_angle_detection()
    elif choice == '6':
        detector = train_multi_angle_model()
        if detector:
            demonstrate_angle_invariance()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main() 