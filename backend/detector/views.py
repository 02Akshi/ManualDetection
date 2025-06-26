from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .manual_predict import predict_image_file
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "model2_backend"))
from manual_detector_with_config import ConfigurableManualDetector
from config import config, apply_preset_config

# Configure your detector
config.model_path = str(Path(__file__).resolve().parent / "model2_backend" / "trained_manual_detector.pkl")
config.threshold = 0.7
config.detection_methods = ['template_matching', 'feature_matching']

import os
print("Model path being used:", config.model_path)
print("Model file exists:", os.path.exists(config.model_path))

# Load the detector once
model2_detector = ConfigurableManualDetector()

class PredictView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response({'error': 'No file uploaded'}, status=400)
        result = predict_image_file(file_obj)
        if result is None:
            return Response({'error': 'Prediction failed'}, status=500)
        # result is now a dict with 'contains_manual' and 'raw_prediction'
        return Response(result)

class PredictModel2View(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response({'error': 'No file uploaded'}, status=400)
        try:
            # Read image from InMemoryUploadedFile
            image = Image.open(file_obj).convert("RGB")
            image_np = np.array(image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            # Run detection (assuming detect_manual_cv exists)
            results = model2_detector.detect_manual_cv(image_cv)
            manual_found = results.get('manual_found', False)
            return Response({
                'contains_manual': manual_found,
                'detection_details': results
            })
        except Exception as e:
            return Response({'error': f'Prediction failed: {e}'}, status=500)