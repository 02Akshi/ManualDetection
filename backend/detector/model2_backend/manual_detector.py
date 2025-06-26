#!/usr/bin/env python3
"""
Manual Detection System with Training
This program detects if a specific manual appears in different images and includes training capabilities.
Compatible with headless environments and avoids Qt/OpenCV display issues.
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from pathlib import Path

class ManualDetector:
    def __init__(self, model_path=None, threshold=0.8):
        """
        Initialize Manual Detector
        
        Args:
            model_path (str): Path to trained model file
            threshold (float): Matching threshold (0.0 to 1.0)
        """
        self.threshold = threshold
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
            'model_version': '1.0'
        }
        
        if model_path:
            self.load_model(model_path)
    
    def add_training_sample(self, image_path, is_positive=True, manual_bbox=None):
        """
        Add a training sample to the detector
        
        Args:
            image_path (str): Path to the training image
            is_positive (bool): True if positive sample, False if negative
            manual_bbox (tuple): Manual bounding box coordinates (x, y, width, height)
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image at {image_path} could not be opened.")
            return
        
        if is_positive:
            self.training_data['positive_samples'].append(image)
            print(f"Added to positive samples: {image_path}")
        else:
            self.training_data['negative_samples'].append(image)
            print(f"Added to negative samples: {image_path}")
        
        # Extract features and update template if positive sample
        if is_positive and manual_bbox:
            x, y, w, h = manual_bbox
            template = image[y:y+h, x:x+w]
            self.train_model(template)
    
    def train_model(self, template_path, positive_samples=None, negative_samples=None):
        """
        Train the detection model with given templates and samples
        
        Args:
            template_path (str): Path to the template image
            positive_samples (list): List of positive sample images
            negative_samples (list): List of negative sample images
        """
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Template image at {template_path} could not be opened.")
            return
        
        # Store template features for matching
        keypoints, features = self.sift.detectAndCompute(template, None)
        self.training_data['template_keypoints'] = keypoints
        self.training_data['template_features'] = features
        self.training_data['template_size'] = template.shape[:2]
        self.training_data['training_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"Model trained with template from {template_path}")
    
    def _optimize_thresholds(self):
        """
        Optimize detection thresholds based on training data
        """
        # Placeholder for threshold optimization logic
        print("Optimizing thresholds...")
    
    def _detect_with_threshold(self, image, threshold):
        """
        Detect objects in the image using the specified threshold
        
        Args:
            image (ndarray): Input image
            threshold (float): Detection threshold
            
        Returns:
            list: Detected keypoints
        """
        keypoints, features = self.sift.detectAndCompute(image, None)
        matches = self.matcher.knnMatch(features, self.training_data['template_features'], k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)
        
        return keypoints, good_matches
    
    def save_model(self, model_path):
        """
        Save the trained model to a file
        
        Args:
            model_path (str): Path to save the model file
        """
        # Serialize the training data to a file
        np.savez_compressed(model_path, **self.training_data)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """
        Load the trained model from a file
        
        Args:
            model_path (str): Path to the model file
        """
        # Deserialize the training data from the file
        data = np.load(model_path, allow_pickle=True)
        self.training_data = {key: data[key] for key in data.files}
        print(f"Model loaded from {model_path}")
    
    def template_matching(self, image):
        """
        Perform template matching on the image
        
        Args:
            image (ndarray): Input image
            
        Returns:
            list: Detected keypoints
        """
        return self._detect_with_threshold(image, self.threshold)
    
    def feature_matching(self, image):
        """
        Perform feature matching on the image
        
        Args:
            image (ndarray): Input image
            
        Returns:
            list: Detected keypoints
        """
        return self._detect_with_threshold(image, self.threshold)
    
    def color_histogram_matching(self, image):
        """
        Perform color histogram matching on the image
        
        Args:
            image (ndarray): Input image
            
        Returns:
            float: Similarity score
        """
        # Compute color histogram
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Compare with template histogram
        similarity = cosine_similarity([hist], [self.training_data['color_histogram']])
        return similarity[0][0]
    
    def detect_manual(self, image_path, methods=['template_matching', 'feature_matching']):
        """
        Detect the manual in the given image using specified methods
        
        Args:
            image_path (str): Path to the image
            methods (list): List of matching methods to use
            
        Returns:
            dict: Detection results
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image at {image_path} could not be opened.")
            return {}
        
        results = {}
        for method in methods:
            if method == 'template_matching':
                keypoints, matches = self.template_matching(image)
                results['template_matching'] = {'keypoints': keypoints, 'matches': matches}
            elif method == 'feature_matching':
                keypoints, matches = self.feature_matching(image)
                results['feature_matching'] = {'keypoints': keypoints, 'matches': matches}
            elif method == 'color_histogram':
                score = self.color_histogram_matching(image)
                results['color_histogram'] = {'score': score}
        
        return results
    
    def draw_matches(self, image, results, output_path=None):
        """
        Draw matches on the image for visualization
        
        Args:
            image (ndarray): Input image
            results (dict): Detection results
            output_path (str): Path to save the output image
        """
        # Draw matches for each method
        for method, result in results.items():
            if method in ['template_matching', 'feature_matching']:
                keypoints = result['keypoints']
                matches = result['matches']
                
                # Draw matches on the image
                image_matches = cv2.drawMatches(image, keypoints, image, keypoints, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imshow(f"Matches - {method}", image_matches)
                
                # Save visualization if output path is provided
                if output_path:
                    cv2.imwrite(f"{output_path}_{method}_matches.jpg", image_matches)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def save_visualization(self, image, results, output_path):
        """
        Save visualization of detection results to a file
        
        Args:
            image (ndarray): Input image
            results (dict): Detection results
            output_path (str): Path to save the visualization image
        """
        # Create a copy of the image for visualization
        image_vis = image.copy()
        
        # Draw bounding boxes for each detected manual
        for result in results:
            x, y, w, h = result['bbox']
            cv2.rectangle(image_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Save the visualization image
        cv2.imwrite(output_path, image_vis)
        print(f"Visualization saved to {output_path}")

def create_training_dataset(template_path, positive_folder, negative_folder, output_file):
    """
    Create a training dataset for the manual detector
    
    Args:
        template_path (str): Path to the template image
        positive_folder (str): Folder containing positive sample images
        negative_folder (str): Folder containing negative sample images
        output_file (str): Path to save the training dataset file
    """
    # Placeholder for dataset creation logic
    print("Creating training dataset...")
    
def main():
    """
    Main function to run the manual detection system
    """
    # Placeholder for main logic
    print("Running Manual Detection System...")
    detector = ManualDetector()
    detector.add_training_sample('path/to/image.jpg', is_positive=True, manual_bbox=(50, 50, 100, 100))
    detector.train_model('path/to/template.jpg')
    detector.save_model('path/to/model_file')
    detector.load_model('path/to/model_file')
    results = detector.detect_manual('path/to/test_image.jpg')
    detector.draw_matches(cv2.imread('path/to/test_image.jpg'), results)

if __name__ == "__main__":
    main()
