#!/usr/bin/env python3
"""
Manual Detection System with Training
This program detects if a specific manual appears in different images and includes training capabilities.
Compatible with headless environments and avoids Qt/OpenCV display issues.
Enhanced with angle-invariant detection for non-top-view images.
"""

import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
from datetime import datetime
from scipy.spatial.distance import cosine
import math

class ManualDetector:
    def __init__(self, model_path=None, threshold=0.8):
        """
        Initialize Manual Detector
        
        Args:
            model_path (str): Path to trained model file
            threshold (float): Matching threshold (0.0 to 1.0)
        """
        self.threshold = threshold
        self.sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04, edgeThreshold=10)
        self.matcher = cv2.BFMatcher()
        
        # Enhanced training data storage for multi-angle detection
        self.training_data = {
            'positive_samples': [],
            'negative_samples': [],
            'template_features': None,
            'template_keypoints': None,
            'template_size': None,
            'color_histogram': None,
            'training_date': None,
            'model_version': '2.0',
            'multi_angle_templates': [],  # Store templates from different angles
            'angle_variations': [],       # Store angle information
            'perspective_matrices': [],   # Store homography matrices
            'edge_features': None,        # Store edge-based features
            'contour_features': None      # Store contour-based features
        }
        
        if model_path:
            self.load_model(model_path)
    
    def add_training_sample(self, image_path, is_positive=True, manual_bbox=None, angle_info=None):
        """
        Add a training sample to improve detection accuracy
        
        Args:
            image_path (str): Path to training image
            is_positive (bool): True if image contains the manual, False otherwise
            manual_bbox (tuple): Bounding box of manual in image (x1, y1, x2, y2)
            angle_info (dict): Information about viewing angle (elevation, azimuth)
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Extract enhanced features
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            
            # Extract edge features for angle-invariant detection
            edges = cv2.Canny(gray, 50, 150)
            edge_keypoints, edge_descriptors = self.sift.detectAndCompute(edges, None)
            
            # Extract contour features
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_features = self._extract_contour_features(contours)
            
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
                'edge_descriptors': edge_descriptors,
                'contour_features': contour_features,
                'color_histogram': hist,
                'bbox': manual_bbox,
                'angle_info': angle_info,
                'added_date': datetime.now().isoformat()
            }
            
            if is_positive:
                self.training_data['positive_samples'].append(sample_data)
                print(f"Added positive sample: {image_path}")
                if angle_info:
                    print(f"  Angle: elevation={angle_info.get('elevation', 'N/A')}, azimuth={angle_info.get('azimuth', 'N/A')}")
            else:
                self.training_data['negative_samples'].append(sample_data)
                print(f"Added negative sample: {image_path}")
                
        except Exception as e:
            print(f"Error adding training sample {image_path}: {e}")
    
    def _extract_contour_features(self, contours):
        """Extract features from contours for angle-invariant detection"""
        features = []
        for contour in contours:
            if len(contour) > 10:  # Only consider significant contours
                # Calculate contour properties
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Get convex hull
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                features.append({
                    'area': area,
                    'perimeter': perimeter,
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'solidity': solidity,
                    'center': (x + w//2, y + h//2)
                })
        
        return features
    
    def add_multi_angle_template(self, image_path, angle_info):
        """
        Add a template from a specific viewing angle
        
        Args:
            image_path (str): Path to template image
            angle_info (dict): Information about viewing angle
                - elevation: angle from horizontal (0-90 degrees)
                - azimuth: rotation around vertical axis (0-360 degrees)
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            
            # Extract edge features
            edges = cv2.Canny(gray, 50, 150)
            edge_keypoints, edge_descriptors = self.sift.detectAndCompute(edges, None)
            
            # Extract contour features
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_features = self._extract_contour_features(contours)
            
            # Calculate color histogram
            hist = cv2.calcHist([image], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            template_data = {
                'image': image,
                'gray': gray,
                'keypoints': keypoints,
                'descriptors': descriptors,
                'edge_descriptors': edge_descriptors,
                'contour_features': contour_features,
                'color_histogram': hist,
                'angle_info': angle_info,
                'size': image.shape
            }
            
            self.training_data['multi_angle_templates'].append(template_data)
            print(f"Added multi-angle template: {image_path}")
            print(f"  Angle: elevation={angle_info.get('elevation', 'N/A')}, azimuth={angle_info.get('azimuth', 'N/A')}")
            
        except Exception as e:
            print(f"Error adding multi-angle template {image_path}: {e}")
    
    def train_model(self, template_path, positive_samples=None, negative_samples=None):
        """
        Train the model using template and training samples
        
        Args:
            template_path (str): Path to the template manual image
            positive_samples (list): List of paths to images containing the manual
            negative_samples (list): List of paths to images without the manual
        """
        print("Starting model training...")
        
        # Load template
        self.template = cv2.imread(template_path)
        if self.template is None:
            raise ValueError(f"Could not read template: {template_path}")
        
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
        
        # Add training samples if provided
        if positive_samples:
            print(f"Adding {len(positive_samples)} positive samples...")
            for sample_path in positive_samples:
                self.add_training_sample(sample_path, is_positive=True)
        
        if negative_samples:
            print(f"Adding {len(negative_samples)} negative samples...")
            for sample_path in negative_samples:
                self.add_training_sample(sample_path, is_positive=False)
        
        # Train the model (calculate optimal thresholds)
        self._optimize_thresholds()
        
        print("Model training completed!")
        print(f"Template size: {self.template.shape}")
        print(f"Template keypoints: {len(self.template_keypoints)}")
        print(f"Positive samples: {len(self.training_data['positive_samples'])}")
        print(f"Negative samples: {len(self.training_data['negative_samples'])}")
    
    def _optimize_thresholds(self):
        """Optimize detection thresholds using training data"""
        print("Optimizing detection thresholds...")
        
        # Test different thresholds on training data
        thresholds = np.arange(0.3, 1.0, 0.05)
        best_f1_score = 0
        best_threshold = 0.8
        
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
        """
        Save the trained model
        
        Args:
            model_path (str): Path to save the model
        """
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
        """
        Load a trained model
        
        Args:
            model_path (str): Path to the model file
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.training_data = model_data['training_data']
            self.template = model_data['template']
            self.template_gray = model_data['template_gray']
            self.template_descriptors = model_data['template_descriptors']
            self.template_hist = model_data['template_hist']
            self.threshold = model_data.get('threshold', 0.8)
            
            # Reconstruct keypoints from serializable format
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
                # Handle old format or missing keypoints
                self.template_keypoints = model_data.get('template_keypoints', [])
            
            print(f"Model loaded from: {model_path}")
            print(f"Training date: {self.training_data.get('training_date', 'Unknown')}")
            print(f"Positive samples: {len(self.training_data['positive_samples'])}")
            print(f"Negative samples: {len(self.training_data['negative_samples'])}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("This might be due to an incompatible model format.")
            print("Please retrain the model.")
            sys.exit(1)
    
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
    
    def multi_angle_template_matching(self, image):
        """Enhanced template matching using multiple angle templates"""
        if not self.training_data['multi_angle_templates']:
            return self.template_matching(image)  # Fallback to original method
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        all_matches = []
        
        for template_data in self.training_data['multi_angle_templates']:
            template_gray = template_data['gray']
            
            # Multi-scale template matching
            scales = [0.5, 0.75, 1.0, 1.25, 1.5]
            for scale in scales:
                if scale != 1.0:
                    width = int(template_gray.shape[1] * scale)
                    height = int(template_gray.shape[0] * scale)
                    resized_template = cv2.resize(template_gray, (width, height))
                else:
                    resized_template = template_gray
                
                result = cv2.matchTemplate(gray, resized_template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= self.threshold)
                
                for pt in zip(*locations[::-1]):
                    confidence = result[pt[1], pt[0]]
                    all_matches.append({
                        'method': 'multi_angle_template',
                        'bbox': (pt[0], pt[1], pt[0] + resized_template.shape[1], pt[1] + resized_template.shape[0]),
                        'confidence': confidence,
                        'center': (pt[0] + resized_template.shape[1]//2, pt[1] + resized_template.shape[0]//2),
                        'angle_info': template_data['angle_info'],
                        'scale': scale
                    })
        
        # Remove overlapping detections
        return self._remove_overlapping_detections(all_matches)
    
    def perspective_invariant_feature_matching(self, image):
        """Enhanced feature matching that handles perspective changes"""
        if not self.training_data['multi_angle_templates']:
            return self.feature_matching(image)  # Fallback to original method
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is None:
            return []
        
        all_matches = []
        
        for template_data in self.training_data['multi_angle_templates']:
            template_descriptors = template_data['descriptors']
            template_keypoints = template_data['keypoints']
            
            if template_descriptors is None:
                continue
            
            # Use FLANN matcher for better performance
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            matches = flann.knnMatch(template_descriptors, descriptors, k=2)
            good_matches = []
            
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:  # More lenient ratio for angle variations
                        good_matches.append(m)
            
            if len(good_matches) < 8:  # Lower threshold for angle variations
                continue
            
            # Find homography with RANSAC
            src_pts = np.float32([template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
            
            if H is None:
                continue
            
            # Transform template corners
            h, w = template_data['gray'].shape
            template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(template_corners, H)
            
            # Calculate bounding box
            x_coords = [pt[0][0] for pt in transformed_corners]
            y_coords = [pt[0][1] for pt in transformed_corners]
            
            x1, x2 = int(min(x_coords)), int(max(x_coords))
            y1, y2 = int(min(y_coords)), int(max(y_coords))
            
            # Calculate confidence based on matches and angle similarity
            base_confidence = min(len(good_matches) / 30.0, 1.0)
            
            # Estimate viewing angle from homography
            estimated_angle = self._estimate_viewing_angle(H)
            
            all_matches.append({
                'method': 'perspective_invariant',
                'bbox': (x1, y1, x2, y2),
                'confidence': base_confidence,
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'matches_count': len(good_matches),
                'angle_info': template_data['angle_info'],
                'estimated_angle': estimated_angle
            })
        
        return self._remove_overlapping_detections(all_matches)
    
    def _estimate_viewing_angle(self, homography_matrix):
        """Estimate viewing angle from homography matrix"""
        try:
            # Extract rotation and translation from homography
            # This is a simplified approach - in practice, you'd need camera calibration
            H = homography_matrix
            
            # Decompose homography to get rotation
            # For simplicity, we'll use the scaling factors
            scale_x = np.sqrt(H[0, 0]**2 + H[0, 1]**2)
            scale_y = np.sqrt(H[1, 0]**2 + H[1, 1]**2)
            
            # Estimate elevation angle from scaling
            # If scale_y < scale_x, it suggests perspective foreshortening
            if scale_y > 0:
                elevation_estimate = math.acos(min(scale_y / scale_x, 1.0)) * 180 / math.pi
            else:
                elevation_estimate = 0
            
            return {
                'elevation_estimate': elevation_estimate,
                'scale_x': scale_x,
                'scale_y': scale_y
            }
        except Exception as e:
            return {'elevation_estimate': 0, 'scale_x': 1, 'scale_y': 1}
    
    def contour_based_detection(self, image):
        """Detect manual using contour features (angle-invariant)"""
        if not self.training_data['multi_angle_templates']:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        image_contour_features = self._extract_contour_features(contours)
        matches = []
        
        for template_data in self.training_data['multi_angle_templates']:
            template_contours = template_data['contour_features']
            
            for img_contour in image_contour_features:
                best_match_score = 0
                
                for template_contour in template_contours:
                    # Compare contour properties
                    area_similarity = 1 - abs(img_contour['area'] - template_contour['area']) / max(img_contour['area'], template_contour['area'])
                    circularity_similarity = 1 - abs(img_contour['circularity'] - template_contour['circularity'])
                    aspect_similarity = 1 - abs(img_contour['aspect_ratio'] - template_contour['aspect_ratio']) / max(img_contour['aspect_ratio'], template_contour['aspect_ratio'])
                    solidity_similarity = 1 - abs(img_contour['solidity'] - template_contour['solidity'])
                    
                    # Weighted combination
                    match_score = (0.3 * area_similarity + 
                                  0.2 * circularity_similarity + 
                                  0.3 * aspect_similarity + 
                                  0.2 * solidity_similarity)
                    
                    if match_score > best_match_score:
                        best_match_score = match_score
                
                if best_match_score > self.threshold:
                    # Get bounding box from contour
                    x, y, w, h = cv2.boundingRect(contours[image_contour_features.index(img_contour)])
                    matches.append({
                        'method': 'contour_based',
                        'bbox': (x, y, x + w, y + h),
                        'confidence': best_match_score,
                        'center': (x + w//2, y + h//2),
                        'angle_info': template_data['angle_info']
                    })
        
        return self._remove_overlapping_detections(matches)
    
    def _remove_overlapping_detections(self, matches, overlap_threshold=0.5):
        """Remove overlapping detections, keeping the highest confidence ones"""
        if not matches:
            return []
        
        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered_matches = []
        for match in matches:
            is_overlapping = False
            
            for existing_match in filtered_matches:
                overlap = self._calculate_overlap(match['bbox'], existing_match['bbox'])
                if overlap > overlap_threshold:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_matches.append(match)
        
        return filtered_matches
    
    def _calculate_overlap(self, bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        return intersection / min(area1, area2)
    
    def detect_manual(self, image_path, methods=['multi_angle_template', 'perspective_invariant', 'contour_based']):
        """Detect manual in an image using enhanced angle-invariant methods"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        print(f"Processing image: {image_path}")
        
        results = {}
        
        # Use enhanced methods by default
        if 'multi_angle_template' in methods:
            results['multi_angle_template'] = self.multi_angle_template_matching(image)
        
        if 'perspective_invariant' in methods:
            results['perspective_invariant'] = self.perspective_invariant_feature_matching(image)
        
        if 'contour_based' in methods:
            results['contour_based'] = self.contour_based_detection(image)
        
        # Fallback to original methods if enhanced methods don't have templates
        if 'template_matching' in methods and not self.training_data['multi_angle_templates']:
            results['template_matching'] = self.template_matching(image)
        
        if 'feature_matching' in methods and not self.training_data['multi_angle_templates']:
            results['feature_matching'] = self.feature_matching(image)
        
        if 'color_histogram' in methods:
            results['color_histogram'] = self.color_histogram_matching(image)
        
        return results, image
    
    def draw_matches(self, image, results, output_path=None):
        """Draw detection results on the image with angle information"""
        result_image = image.copy()
        
        colors = {
            'template_matching': (0, 255, 0),
            'feature_matching': (255, 0, 0),
            'color_histogram': (0, 0, 255),
            'multi_angle_template': (0, 255, 255),
            'perspective_invariant': (255, 0, 255),
            'contour_based': (255, 255, 0)
        }
        
        for method, matches in results.items():
            color = colors.get(method, (255, 255, 255))
            
            for i, match in enumerate(matches):
                x1, y1, x2, y2 = match['bbox']
                confidence = match['confidence']
                
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                
                # Create enhanced label with angle information
                label = f"{method}: {confidence:.2f}"
                if 'matches_count' in match:
                    label += f" ({match['matches_count']} matches)"
                
                # Add angle information if available
                if 'angle_info' in match and match['angle_info']:
                    angle_info = match['angle_info']
                    if 'elevation' in angle_info and 'azimuth' in angle_info:
                        label += f" | Elev: {angle_info['elevation']}° Az: {angle_info['azimuth']}°"
                
                if 'estimated_angle' in match:
                    est_angle = match['estimated_angle']
                    if 'elevation_estimate' in est_angle:
                        label += f" | Est: {est_angle['elevation_estimate']:.1f}°"
                
                if 'scale' in match:
                    label += f" | Scale: {match['scale']:.2f}"
                
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
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
                    0.5, 
                    (255, 255, 255), 
                    2
                )
        
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"Output saved to: {output_path}")
        
        return result_image
    
    def save_visualization(self, image, results, output_path):
        """Save visualization using matplotlib (headless)"""
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

    def estimate_angle_from_image(self, image_path):
        """
        Automatically estimate viewing angle from an image
        This uses computer vision techniques to estimate the viewing angle
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            dict: Estimated angle information {'elevation': float, 'azimuth': float}
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'elevation': 0, 'azimuth': 0}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Estimate from aspect ratio changes
            # If the manual appears more compressed vertically, it's likely viewed from an angle
            h, w = gray.shape
            aspect_ratio = w / h
            
            # Method 2: Estimate from edge distribution
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (likely the manual)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w_contour, h_contour = cv2.boundingRect(largest_contour)
                contour_aspect = w_contour / h_contour if h_contour > 0 else 1
                
                # Estimate elevation based on aspect ratio compression
                # A compressed aspect ratio suggests viewing from an angle
                if contour_aspect < 0.7:  # Significantly compressed
                    estimated_elevation = 60
                elif contour_aspect < 0.85:  # Moderately compressed
                    estimated_elevation = 45
                elif contour_aspect < 0.95:  # Slightly compressed
                    estimated_elevation = 30
                else:  # Close to normal aspect ratio
                    estimated_elevation = 0
            else:
                estimated_elevation = 0
            
            # Method 3: Estimate from perspective lines (if available)
            # Look for converging lines that might indicate perspective
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            if lines is not None and len(lines) > 5:
                # If we have many lines, check for perspective
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angles.append(theta)
                
                # Check if lines are converging (perspective)
                angle_variance = np.var(angles)
                if angle_variance > 0.5:  # High variance suggests perspective
                    estimated_elevation = max(estimated_elevation, 30)
            
            # For azimuth, we'll default to 0 (front view)
            # In practice, this is harder to estimate without additional context
            estimated_azimuth = 0
            
            return {
                'elevation': estimated_elevation,
                'azimuth': estimated_azimuth,
                'confidence': 0.6,  # Medium confidence for automatic estimation
                'method': 'automatic'
            }
            
        except Exception as e:
            print(f"Error estimating angle for {image_path}: {e}")
            return {'elevation': 0, 'azimuth': 0, 'confidence': 0.0, 'method': 'error'}

    def add_training_sample_auto_angle(self, image_path, is_positive=True, manual_bbox=None):
        """
        Add a training sample with automatic angle estimation
        
        Args:
            image_path (str): Path to training image
            is_positive (bool): True if image contains the manual, False otherwise
            manual_bbox (tuple): Bounding box of manual in image (x1, y1, x2, y2)
        """
        # Automatically estimate angle
        estimated_angle = self.estimate_angle_from_image(image_path)
        
        print(f"Auto-estimated angle for {os.path.basename(image_path)}: "
              f"Elevation={estimated_angle['elevation']}°, Azimuth={estimated_angle['azimuth']}° "
              f"(confidence: {estimated_angle.get('confidence', 0):.2f})")
        
        # Add the sample with estimated angle
        self.add_training_sample(image_path, is_positive, manual_bbox, estimated_angle)

    def add_multi_angle_template_auto(self, image_path):
        """
        Add a template with automatic angle estimation
        
        Args:
            image_path (str): Path to template image
        """
        # Automatically estimate angle
        estimated_angle = self.estimate_angle_from_image(image_path)
        
        print(f"Auto-estimated angle for template {os.path.basename(image_path)}: "
              f"Elevation={estimated_angle['elevation']}°, Azimuth={estimated_angle['azimuth']}° "
              f"(confidence: {estimated_angle.get('confidence', 0):.2f})")
        
        # Add the template with estimated angle
        self.add_multi_angle_template(image_path, estimated_angle)

    def cluster_training_samples_by_angle(self):
        """
        Automatically group training samples by similar viewing angles
        This helps organize samples when angles weren't provided
        """
        if not self.training_data['positive_samples']:
            return
        
        print("Clustering training samples by estimated angles...")
        
        # Extract angle information from all positive samples
        angle_data = []
        for i, sample in enumerate(self.training_data['positive_samples']):
            if 'angle_info' in sample and sample['angle_info']:
                angle_info = sample['angle_info']
                elevation = angle_info.get('elevation', 0)
                azimuth = angle_info.get('azimuth', 0)
                angle_data.append({
                    'index': i,
                    'elevation': elevation,
                    'azimuth': azimuth,
                    'image_path': sample['image_path']
                })
        
        if not angle_data:
            print("No angle information available for clustering.")
            return
        
        # Simple clustering by elevation ranges
        elevation_clusters = {
            'top_view': (0, 15),      # 0-15 degrees
            'slight_angle': (15, 35), # 15-35 degrees
            'medium_angle': (35, 55), # 35-55 degrees
            'steep_angle': (55, 75),  # 55-75 degrees
            'side_view': (75, 90)     # 75-90 degrees
        }
        
        cluster_results = {}
        for cluster_name, (min_elev, max_elev) in elevation_clusters.items():
            cluster_samples = [data for data in angle_data 
                             if min_elev <= data['elevation'] < max_elev]
            if cluster_samples:
                cluster_results[cluster_name] = cluster_samples
                print(f"{cluster_name}: {len(cluster_samples)} samples "
                      f"(elevation {min_elev}-{max_elev}°)")
        
        return cluster_results


def create_training_dataset(template_path, positive_folder, negative_folder, output_file):
    """
    Create a training dataset configuration file
    
    Args:
        template_path (str): Path to template manual
        positive_folder (str): Folder with images containing the manual
        negative_folder (str): Folder with images without the manual
        output_file (str): Path to save training configuration
    """
    config = {
        'template_path': template_path,
        'positive_samples': [],
        'negative_samples': [],
        'created_date': datetime.now().isoformat()
    }
    
    # Add positive samples
    if os.path.exists(positive_folder):
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            config['positive_samples'].extend(
                [str(f) for f in Path(positive_folder).glob(f"*{ext}")]
            )
            config['positive_samples'].extend(
                [str(f) for f in Path(positive_folder).glob(f"*{ext.upper()}")]
            )
    
    # Add negative samples
    if os.path.exists(negative_folder):
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            config['negative_samples'].extend(
                [str(f) for f in Path(negative_folder).glob(f"*{ext}")]
            )
            config['negative_samples'].extend(
                [str(f) for f in Path(negative_folder).glob(f"*{ext.upper()}")]
            )
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training dataset configuration saved to: {output_file}")
    print(f"Positive samples: {len(config['positive_samples'])}")
    print(f"Negative samples: {len(config['negative_samples'])}")


def main():
    parser = argparse.ArgumentParser(description="Manual Detection System with Training")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--template", help="Path to template manual image")
    parser.add_argument("--positive", help="Folder with positive training samples")
    parser.add_argument("--negative", help="Folder with negative training samples")
    parser.add_argument("--model", help="Path to save/load model file")
    parser.add_argument("--input", help="Path to input image or folder")
    parser.add_argument("-o", "--output", help="Path to output file or folder")
    parser.add_argument("-t", "--threshold", type=float, default=0.8, help="Detection threshold")
    parser.add_argument("--create-dataset", help="Create training dataset configuration")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
    
    args = parser.parse_args()
    
    if args.train:
        # Training mode
        if not args.template or not args.model:
            print("Error: --template and --model are required for training")
            sys.exit(1)
        
        detector = ManualDetector(threshold=args.threshold)
        
        positive_samples = []
        negative_samples = []
        
        if args.positive:
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                positive_samples.extend(Path(args.positive).glob(f"*{ext}"))
                positive_samples.extend(Path(args.positive).glob(f"*{ext.upper()}"))
        
        if args.negative:
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                negative_samples.extend(Path(args.negative).glob(f"*{ext}"))
                negative_samples.extend(Path(args.negative).glob(f"*{ext.upper()}"))
        
        detector.train_model(
            args.template,
            [str(f) for f in positive_samples],
            [str(f) for f in negative_samples]
        )
        
        detector.save_model(args.model)
        
    elif args.create_dataset:
        # Create training dataset configuration
        if not args.template or not args.positive or not args.negative:
            print("Error: --template, --positive, and --negative are required for dataset creation")
            sys.exit(1)
        
        create_training_dataset(args.template, args.positive, args.negative, args.create_dataset)
        
    else:
        # Detection mode
        if not args.model or not args.input:
            print("Error: --model and --input are required for detection")
            sys.exit(1)
        
        detector = ManualDetector(args.model, args.threshold)
        
        if os.path.isdir(args.input):
            # Batch processing
            os.makedirs(args.output, exist_ok=True) if args.output else None
            
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                for image_file in Path(args.input).glob(f"*{ext}"):
                    try:
                        results, image = detector.detect_manual(str(image_file))
                        
                        if args.output:
                            # Save result image
                            output_path = Path(args.output) / f"result_{image_file.name}"
                            detector.draw_matches(image, results, str(output_path))
                            
                            # Save visualization
                            viz_path = Path(args.output) / f"viz_{image_file.name}.png"
                            detector.save_visualization(image, results, str(viz_path))
                        
                        manual_found = any(len(matches) > 0 for matches in results.values())
                        print(f"{image_file.name}: Manual found = {manual_found}")
                        
                    except Exception as e:
                        print(f"Error processing {image_file}: {e}")
        else:
            # Single image processing
            results, image = detector.detect_manual(args.input)
            
            if args.output:
                # Save result image
                detector.draw_matches(image, results, args.output)
                
                # Save visualization
                viz_path = args.output.replace('.jpg', '_viz.png').replace('.png', '_viz.png')
                detector.save_visualization(image, results, viz_path)
            
            manual_found = any(len(matches) > 0 for matches in results.values())
            print(f"Manual found: {manual_found}")
            
            # Print detailed results
            for method, matches in results.items():
                print(f"  {method}: {len(matches)} matches")
                for match in matches:
                    print(f"    Confidence: {match['confidence']:.3f}")


if __name__ == "__main__":
    main() 