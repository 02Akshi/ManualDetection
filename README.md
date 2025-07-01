# Manual Detection System

This Python program specializes in detecting if a specific manual appears in different images. Unlike general object detection, this system is designed to find exact or similar instances of a particular manual document.

# Project Setup Guide

## Contents

- `backend/`  
  Django backend API, model, training script, and dataset for retraining.
- `basket-manual-detector/frontend/`  
  React frontend for testing the API.

---

## Backend Setup

1. **Install Python 3.8+ and pip** (if not already installed).

2. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

3. **(Recommended) Create and activate a virtual environment:**
   ```bash
    sudo apt install virtualenv
    mkdir /opt/model-detector
    virtualenv -p python3 /opt/model-detector/plugins
    source /opt/model-detector/plugins/bin/activate
   ```
4. **Clone the project:** 
    sudo git clone <your-repo-url> /opt/model-detector/
    cd /opt/model-detector/backend
    
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **(Optional) Create a Django superuser:**
   ```bash
   python manage.py createsuperuser
   ```

6. **Run the backend server:**
   ```bash
   python manage.py runserver 0.0.0.0:8000
   ```
   The API will be available at `http://localhost:8000/` (or your server’s IP).

7. **API Endpoints:**
   - `POST /predict/` — Upload an image for prediction (requires authentication).
   - `POST /login/` — Login endpoint.
   - `POST /logout/` — Logout endpoint.

---

## Retraining the Model

- The training script is `backend/manual_classifier.py`.
- The dataset is in `backend/dataset_split/` (with `train/`, `val/`, `test/` subfolders).
- To retrain:
  1. Edit or run `manual_classifier.py` as needed.
  2. The trained model will be saved as `manual_classifier_mobilenetv2.h5` in `backend/`.

---

## Frontend Setup (for testing)

1. **Navigate to the frontend directory:**
   ```bash
   cd basket-manual-detector/frontend
   ```

2. **Install Node.js and npm** (if not already installed).

3. **Install frontend dependencies:**
   ```bash
   npm install
   ```

4. **Start the frontend development server:**
   ```bash
   npm run dev
   ```
   The frontend will be available at `http://localhost:5173/` (or as shown in the terminal).

5. **Configure the frontend to point to your backend API if needed (see frontend README).**

---

## Notes

- Make sure `manual_classifier_mobilenetv2.h5` is present in `backend/` for predictions.
- If you want to use your own dataset, replace the contents of `dataset_split/`.
- For production deployment, use a production-ready server (e.g., Gunicorn for Django) and configure static/media files.
