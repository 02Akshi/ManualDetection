#!/usr/bin/env python3
"""
Enhanced Manual Detection System with Configuration
This version uses a configuration system for easy setup and customization.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
from datetime import datetime

# Import configuration
from config import config, load_config_from_file, save_config_to_file, validate_config, print_config, apply_preset_config

class ConfigurableManualDetector:

    def __init__(self, config_file=None, preset=None):
        """
        Initialize detector with configuration
        
        Args:
            config_file (str): Path to configuration file
            preset (str): Preset configuration name
        """
        # Load configuration
        if config_file:
            load_config_from_file(config_file)
        
        if preset:
            apply_preset_config(preset)
        
        # Initialize detector with config settings
        self.threshold = config.threshold
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        
        # Training data storage
        self.training_data = {
            'positive_samples': [],
            'negative_samples': [],
            'template_features': None,
            'template_keypoints': None,
            'template_size': None,
            'color_histogram': None,
            'training_date': None,
            'model_version': '2.0'
        }
        
        # Load model if exists
        if os.path.exists(config.model_path):
            self.load_model(config.model_path)

    def detect_manual_cv(self, image):
        """
        Detect manual in an image (OpenCV format) using configuration settings.
        """
        # Ensure model is loaded
        if not hasattr(self, 'template') or self.template is None:
            raise ValueError("Model not loaded or trained. Please train the model or provide a valid model file.")
        
        # Resize image if configured
        if config.resize_images:
            h, w = image.shape[:2]
            max_w, max_h = config.max_image_size
            if w > max_w or h > max_h:
                scale = min(max_w / w, max_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h))

        results = {}
        for method in config.detection_methods:
            if method == 'template_matching':
                results['template_matching'] = self.template_matching(image)
            elif method == 'feature_matching':
                results['feature_matching'] = self.feature_matching(image)
            elif method == 'color_histogram':
                results['color_histogram'] = self.color_histogram_matching(image)

        # Filter results based on configuration
        filtered_results = {}
        for method, matches in results.items():
            filtered_matches = [m for m in matches if m['confidence'] >= config.min_confidence]
            if len(filtered_matches) > config.max_detections:
                filtered_matches = sorted(filtered_matches, key=lambda x: x['confidence'], reverse=True)[:config.max_detections]
            filtered_results[method] = filtered_matches

        manual_found = any(len(matches) > 0 for matches in filtered_results.values())
        return {
            'manual_found': manual_found,
            'matches': filtered_results
        }
    
    def train_model(self):
        """Train the model using configuration settings"""
        print("Starting model training with configuration...")
        print_config()
        
        # Validate configuration
        errors, warnings = validate_config()
        
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  ✗ {error}")
            print("Please fix these errors before training.")
            return False
        
        if warnings:
            print("Configuration warnings:")
            for warning in warnings:
                print(f"  ⚠ {warning}")
        
        # Load template
        self.template = cv2.imread(config.template_path)
        if self.template is None:
            print(f"Error: Could not read template from {config.template_path}")
            return False
        
        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        
        # Extract template features
        self.template_keypoints, self.template_descriptors = self.sift.detectAndCompute(
            self.template_gray, None
        )
        
        # Calculate template color histogram
        self.template_hist = cv2.calcHist([self.template], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        self.template_hist = cv2.normalize(self.template_hist, self.template_hist).flatten()
        
        # Store template data
        self.training_data['template_features'] = self.template_descriptors
        self.training_data['template_size'] = self.template.shape
        self.training_data['color_histogram'] = self.template_hist
        self.training_data['training_date'] = datetime.now().isoformat()
        
        # Collect training samples
        positive_samples = []
        negative_samples = []
        
        if os.path.exists(config.positive_samples_folder):
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                positive_samples.extend(Path(config.positive_samples_folder).glob(f"*{ext}"))
                positive_samples.extend(Path(config.positive_samples_folder).glob(f"*{ext.upper()}"))
        
        if os.path.exists(config.negative_samples_folder):
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                negative_samples.extend(Path(config.negative_samples_folder).glob(f"*{ext}"))
                negative_samples.extend(Path(config.negative_samples_folder).glob(f"*{ext.upper()}"))
        
        print(f"Found {len(positive_samples)} positive samples")
        print(f"Found {len(negative_samples)} negative samples")
        
        # Add training samples
        for sample_path in positive_samples:
            self.add_training_sample(str(sample_path), is_positive=True)
        
        for sample_path in negative_samples:
            self.add_training_sample(str(sample_path), is_positive=False)
        
        # Optimize thresholds
        self._optimize_thresholds()
        
        # Save model
        self.save_model(config.model_path)
        
        print("Model training completed!")
        return True
    
    def add_training_sample(self, image_path, is_positive=True, manual_bbox=None):
        """Add a training sample"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Resize image if configured
            if config.resize_images:
                h, w = image.shape[:2]
                max_w, max_h = config.max_image_size
                if w > max_w or h > max_h:
                    scale = min(max_w / w, max_h / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = cv2.resize(image, (new_w, new_h))
            
            # Extract features
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            
            # Convert keypoints to serializable format
            serializable_keypoints = []
            if keypoints is not None:
                for kp in keypoints:
                    serializable_keypoints.append({
                        'pt': kp.pt,
                        'size': kp.size,
                        'angle': kp.angle,
                        'response': kp.response,
                        'octave': kp.octave,
                        'class_id': kp.class_id
                    })
            
            # Calculate color histogram
            hist = cv2.calcHist([image], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            sample_data = {
                'image_path': image_path,
                'image_size': image.shape,
                'keypoints_serializable': serializable_keypoints,
                'descriptors': descriptors,
                'color_histogram': hist,
                'bbox': manual_bbox,
                'added_date': datetime.now().isoformat()
            }
            
            if is_positive:
                self.training_data['positive_samples'].append(sample_data)
                print(f"Added positive sample: {image_path}")
            else:
                self.training_data['negative_samples'].append(sample_data)
                print(f"Added negative sample: {image_path}")
                
        except Exception as e:
            print(f"Error adding training sample {image_path}: {e}")
    
    def _optimize_thresholds(self):
        """Optimize detection thresholds"""
        print("Optimizing detection thresholds...")
        
        thresholds = np.arange(0.3, 1.0, 0.05)
        best_f1_score = 0
        best_threshold = config.threshold
        
        for threshold in thresholds:
            tp, fp, tn, fn = 0, 0, 0, 0
            
            # Test on positive samples
            for sample in self.training_data['positive_samples']:
                image = cv2.imread(sample['image_path'])
                results, _ = self._detect_with_threshold(image, threshold)
                if len(results) > 0:
                    tp += 1
                else:
                    fn += 1
            
            # Test on negative samples
            for sample in self.training_data['negative_samples']:
                image = cv2.imread(sample['image_path'])
                results, _ = self._detect_with_threshold(image, threshold)
                if len(results) > 0:
                    fp += 1
                else:
                    tn += 1
            
            # Calculate F1 score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_threshold = threshold
        
        self.threshold = best_threshold
        config.threshold = best_threshold  # Update config
        print(f"Optimal threshold: {best_threshold:.2f} (F1 score: {best_f1_score:.3f})")
    
    def _detect_with_threshold(self, image, threshold):
        """Internal method for threshold testing"""
        results = {}
        
        # Template matching
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        
        template_matches = []
        for pt in zip(*locations[::-1]):
            confidence = result[pt[1], pt[0]]
            template_matches.append({
                'bbox': (pt[0], pt[1], pt[0] + self.template.shape[1], pt[1] + self.template.shape[0]),
                'confidence': confidence
            })
        
        results['template_matching'] = template_matches
        return results, image
    
    def save_model(self, model_path):
        """Save the trained model"""
        try:
            # Convert keypoints to serializable format
            serializable_keypoints = []
            if self.template_keypoints is not None:
                for kp in self.template_keypoints:
                    serializable_keypoints.append({
                        'pt': kp.pt,
                        'size': kp.size,
                        'angle': kp.angle,
                        'response': kp.response,
                        'octave': kp.octave,
                        'class_id': kp.class_id
                    })
            
            model_data = {
                'training_data': self.training_data,
                'template': self.template,
                'template_gray': self.template_gray,
                'template_keypoints_serializable': serializable_keypoints,
                'template_descriptors': self.template_descriptors,
                'template_hist': self.template_hist,
                'threshold': self.threshold
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"Model saved to: {model_path}")
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, model_path):
        """Load a trained model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.training_data = model_data['training_data']
            self.template = model_data['template']
            self.template_gray = model_data['template_gray']
            self.template_descriptors = model_data['template_descriptors']
            self.template_hist = model_data['template_hist']
            self.threshold = model_data.get('threshold', config.threshold)
            
            # Reconstruct keypoints
            if 'template_keypoints_serializable' in model_data:
                serializable_keypoints = model_data['template_keypoints_serializable']
                self.template_keypoints = []
                for kp_data in serializable_keypoints:
                    kp = cv2.KeyPoint(
                        x=kp_data['pt'][0],
                        y=kp_data['pt'][1],
                        size=kp_data['size'],
                        angle=kp_data['angle'],
                        response=kp_data['response'],
                        octave=kp_data['octave'],
                        class_id=kp_data['class_id']
                    )
                    self.template_keypoints.append(kp)
            else:
                self.template_keypoints = model_data.get('template_keypoints', [])
            
            print(f"Model loaded from: {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def detect_manual(self, image_path):
        """Detect manual in an image using configuration settings"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Resize image if configured
        if config.resize_images:
            h, w = image.shape[:2]
            max_w, max_h = config.max_image_size
            if w > max_w or h > max_h:
                scale = min(max_w / w, max_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h))
        
        print(f"Processing image: {image_path}")
        
        results = {}
        
        # Use configured detection methods
        for method in config.detection_methods:
            if method == 'template_matching':
                results['template_matching'] = self.template_matching(image)
            elif method == 'feature_matching':
                results['feature_matching'] = self.feature_matching(image)
            elif method == 'color_histogram':
                results['color_histogram'] = self.color_histogram_matching(image)
        
        # Filter results based on configuration
        filtered_results = {}
        for method, matches in results.items():
            # Filter by confidence
            filtered_matches = [m for m in matches if m['confidence'] >= config.min_confidence]
            
            # Limit number of detections
            if len(filtered_matches) > config.max_detections:
                filtered_matches = sorted(filtered_matches, key=lambda x: x['confidence'], reverse=True)[:config.max_detections]
            
            filtered_results[method] = filtered_matches
        
        return filtered_results, image
    
    def template_matching(self, image):
        """Template matching detection"""
        if self.template is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= self.threshold)
        
        matches = []
        for pt in zip(*locations[::-1]):
            confidence = result[pt[1], pt[0]]
            matches.append({
                'method': 'template_matching',
                'bbox': (pt[0], pt[1], pt[0] + self.template.shape[1], pt[1] + self.template.shape[0]),
                'confidence': confidence,
                'center': (pt[0] + self.template.shape[1]//2, pt[1] + self.template.shape[0]//2)
            })
        
        return matches
    
    def feature_matching(self, image):
        """Feature matching detection"""
        if self.template_descriptors is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is None:
            return []
        
        matches = self.matcher.knnMatch(self.template_descriptors, descriptors, k=2)
        good_matches = []
        
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            return []
        
        src_pts = np.float32([self.template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return []
        
        h, w = self.template_gray.shape
        template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(template_corners, H)
        
        x_coords = [pt[0][0] for pt in transformed_corners]
        y_coords = [pt[0][1] for pt in transformed_corners]
        
        x1, x2 = int(min(x_coords)), int(max(x_coords))
        y1, y2 = int(min(y_coords)), int(max(y_coords))
        
        confidence = min(len(good_matches) / 50.0, 1.0)
        
        return [{
            'method': 'feature_matching',
            'bbox': (x1, y1, x2, y2),
            'confidence': confidence,
            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
            'matches_count': len(good_matches)
        }]
    
    def color_histogram_matching(self, image):
        """Color histogram matching detection"""
        if self.template_hist is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        window_size = self.template.shape[:2]
        step_size = 20
        matches = []
        
        for y in range(0, image.shape[0] - window_size[0], step_size):
            for x in range(0, image.shape[1] - window_size[1], step_size):
                window = image[y:y + window_size[0], x:x + window_size[1]]
                window_hist = cv2.calcHist([window], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
                window_hist = cv2.normalize(window_hist, window_hist).flatten()
                
                similarity = cv2.compareHist(self.template_hist, window_hist, cv2.HISTCMP_CORREL)
                
                if similarity > self.threshold:
                    matches.append({
                        'method': 'color_histogram',
                        'bbox': (x, y, x + window_size[1], y + window_size[0]),
                        'confidence': similarity,
                        'center': (x + window_size[1]//2, y + window_size[0]//2)
                    })
        
        return matches
    
    def draw_matches(self, image, results, output_path=None):
        """Draw detection results using configuration settings"""
        result_image = image.copy()
        
        for method, matches in results.items():
            color = config.colors.get(method, (255, 255, 255))
            
            for i, match in enumerate(matches):
                x1, y1, x2, y2 = match['bbox']
                confidence = match['confidence']
                
                # Draw bounding box
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, config.bounding_box_thickness)
                
                # Create label
                label_parts = []
                if config.show_detection_methods:
                    label_parts.append(method)
                if config.show_confidence_scores:
                    label_parts.append(f"{confidence:.2f}")
                
                label = ": ".join(label_parts)
                
                if 'matches_count' in match:
                    label += f" ({match['matches_count']} matches)"
                
                # Draw label
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, config.text_scale, 2
                )
                
                cv2.rectangle(
                    result_image, 
                    (x1, y1 - text_height - baseline - 10), 
                    (x1 + text_width, y1), 
                    color, 
                    -1
                )
                
                cv2.putText(
                    result_image, 
                    label, 
                    (x1, y1 - baseline - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    config.text_scale, 
                    (255, 255, 255), 
                    2
                )
        
        if output_path and config.save_detection_images:
            cv2.imwrite(output_path, result_image)
            print(f"Detection image saved to: {output_path}")
        
        return result_image
    
    def save_visualization(self, image, results, output_path):
        """Save visualization using configuration settings"""
        if not config.save_visualizations:
            return
        
        plt.figure(figsize=(15, 8))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        # Result image
        plt.subplot(1, 2, 2)
        result_image = self.draw_matches(image, results)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title("Detection Results")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {output_path}")
    
    def process_batch(self):
        """Process all images in the configured input folder"""
        print("Starting batch processing...")
        print_config()
        
        # Validate configuration
        errors, warnings = validate_config()
        
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  ✗ {error}")
            return False
        
        # Ensure output folder exists
        os.makedirs(config.output_folder, exist_ok=True)
        
        # Process images
        if os.path.isfile(config.input_path):
            # Single file
            self.process_single_image(config.input_path)
        else:
            # Folder
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(Path(config.input_path).glob(f"*{ext}"))
                image_files.extend(Path(config.input_path).glob(f"*{ext.upper()}"))
            
            print(f"Found {len(image_files)} images to process")
            
            for image_file in image_files:
                self.process_single_image(str(image_file))
        
        print("Batch processing completed!")
        return True
    
    def process_single_image(self, image_path):
        """Process a single image"""
        try:
            print(f"\nProcessing: {Path(image_path).name}")
            
            # Detect manual
            results, image = self.detect_manual(image_path)
            
            # Check if manual was found
            manual_found = any(len(matches) > 0 for matches in results.values())
            
            # Save results
            if config.save_detection_images:
                output_path = Path(config.output_folder) / f"detected_{Path(image_path).name}"
                self.draw_matches(image, results, str(output_path))
            
            if config.save_visualizations:
                viz_path = Path(config.output_folder) / f"viz_{Path(image_path).stem}.png"
                self.save_visualization(image, results, str(viz_path))
            
            # Print results
            print(f"  Manual found: {manual_found}")
            for method, matches in results.items():
                print(f"    {method}: {len(matches)} matches")
                for match in matches:
                    print(f"      Confidence: {match['confidence']:.3f}")
            
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")


def main():
    """Main function with configuration-based operation"""
    print("Configurable Manual Detection System")
    print("=" * 50)
    
    # Load configuration
    load_config_from_file()
    
    # Create detector
    detector = ConfigurableManualDetector()
    
    while True:
        print("\nConfiguration Menu:")
        print("1. Show current configuration")
        print("2. Train model")
        print("3. Process single image")
        print("4. Process batch (folder)")
        print("5. Apply preset configuration")
        print("6. Save configuration")
        print("7. Validate configuration")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            print_config()
        elif choice == '2':
            detector.train_model()
        elif choice == '3':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                detector.process_single_image(image_path)
            else:
                print(f"Image not found: {image_path}")
        elif choice == '4':
            detector.process_batch()
        elif choice == '5':
            print("Available presets: high_accuracy, fast_processing, video_processing")
            preset = input("Enter preset name: ").strip()
            apply_preset_config(preset)
        elif choice == '6':
            save_config_to_file()
        elif choice == '7':
            errors, warnings = validate_config()
            if errors:
                print("Errors:")
                for error in errors:
                    print(f"  ✗ {error}")
            if warnings:
                print("Warnings:")
                for warning in warnings:
                    print(f"  ⚠ {warning}")
            if not errors and not warnings:
                print("✓ Configuration is valid!")
        elif choice == '8':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main() 