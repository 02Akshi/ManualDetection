# filepath: /home/akshita-bindal/Desktop/new_manual/website/backend/detector/urls.py
from django.urls import path
from .views import PredictView

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
]