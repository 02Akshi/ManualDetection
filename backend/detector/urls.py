# filepath: /home/akshita-bindal/Desktop/new_manual/website/backend/detector/urls.py
from django.urls import path
from .views import PredictView, PredictModel2View

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
    path('predict_model2/', PredictModel2View.as_view(), name='predict_model2'),
]