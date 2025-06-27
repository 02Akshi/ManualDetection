from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .manual_predict import predict_image_file
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

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