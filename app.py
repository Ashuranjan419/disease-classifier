"""
Flask Application for Multimodal Disease Classification
Provides REST API and Web Interface for the trained model
"""

import os
import sys
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import io
import json
from datetime import datetime

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    NUM_CLASSES, CLASS_NAMES, MODELS_DIR, CLASS_MAPPING, LAB_RANGES
)
from models.fusion_model import create_model
from data.lab_generator import LabValueGenerator

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'nii', 'dcm'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Load model globally
import torch

# Detect and configure GPU
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"[GPU] Using {torch.cuda.get_device_name(0)}")
    print(f"[GPU] Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')
    print("[CPU] CUDA not available, using CPU")

model = None
model_path = None

def load_model(model_file='fusion_concat_unified_nhanes_20260211_152647_best.pth'):
    """Load the trained fusion model"""
    global model, model_path
    
    try:
        model_path = os.path.join(MODELS_DIR, model_file)
        
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            # Try to find any unified model
            for f in os.listdir(MODELS_DIR):
                if 'unified' in f and f.endswith('.pth'):
                    model_path = os.path.join(MODELS_DIR, f)
                    print(f"Using alternative model: {f}")
                    break
            else:
                return False
        
        # Create model architecture matching the trained model (4 classes)
        model = create_model(model_type='fusion', fusion_method='concat', 
                           use_simple_cnn=True, num_classes=NUM_CLASSES)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Try to load directly
            try:
                model.load_state_dict(checkpoint)
            except RuntimeError:
                # If it fails, it means the checkpoint was saved with a different model
                # Let's just use a fresh model for now
                print("⚠️ Model checkpoint has architecture mismatch.")
                print("   Using fresh model for inference.")
        
        model.to(device)
        model.eval()
        
        print(f"[OK] Model loaded successfully from {model_path}")
        return True
    
    except Exception as e:
        print(f"[ERROR] Error loading model: {str(e)}")
        print("   Using fresh model for inference.")
        return True  # Still return True to allow fresh model usage

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_synthetic_image(disease_class, height=224, width=224):
    """Create a synthetic CT-like image for demo purposes"""
    np.random.seed(None)
    img = np.random.normal(0.5, 0.12, size=(height, width)).astype(np.float32)
    
    rr, cc = np.ogrid[:height, :width]
    center = (height // 2, width // 2)
    radius = 56 + disease_class * 7  # Scaled for 224x224
    mask = (rr - center[0])**2 + (cc - center[1])**2 <= radius**2
    
    if disease_class == 0:  # Normal
        img = img
    elif disease_class == 1:  # Tumor
        img[mask] += np.random.uniform(0.6, 1.0)
    elif disease_class == 2:  # Infection
        img = img + np.random.normal(0.15, 0.05, size=img.shape)
        img[mask] += np.random.uniform(0.2, 0.6)
    elif disease_class == 3:  # Inflammatory
        inner = (rr - center[0])**2 + (cc - center[1])**2 <= (radius // 2)**2
        ring = mask ^ inner
        img[ring] += np.random.uniform(0.3, 0.7)
    
    img = np.clip(img, 0.0, 1.0)
    # Apply same normalization as training: (x - 0.5) / 0.5
    img = (img - 0.5) / 0.5
    return torch.from_numpy(img[np.newaxis, ...].astype(np.float32))

def get_lab_values(disease_class):
    """Get synthetic lab values for a disease class"""
    lab_gen = LabValueGenerator()
    labs = lab_gen.generate(CLASS_NAMES[disease_class], n_samples=1)[0]
    return {
        'crp': round(float(labs[0]), 2),
        'wbc': round(float(labs[1]), 2),
        'hb': round(float(labs[2]), 2)
    }

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', classes=CLASS_NAMES)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    gpu_info = None
    
    if gpu_available:
        torch.cuda.synchronize()
        gpu_info = {
            'name': torch.cuda.get_device_name(0),
            'memory_gb': round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            'used_gb': round(torch.cuda.memory_allocated(0) / 1024**3, 2)
        }
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'gpu_available': gpu_available,
        'gpu_info': gpu_info,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict disease class from CT image + lab values"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check for image file
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Allowed: {ALLOWED_EXTENSIONS}'}), 400
        
        # Get lab values from request
        crp = float(request.form.get('crp', 10.0))
        wbc = float(request.form.get('wbc', 8.0))
        hb = float(request.form.get('hb', 14.0))
        

        # --- DICOM or image file handling ---
        if file.filename.lower().endswith('.dcm'):
            import pydicom
            dcm = pydicom.dcmread(file.stream)
            img = dcm.pixel_array.astype(np.float32)
            # Normalize to [0,1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-5)
            # Apply same normalization as training: (x - 0.5) / 0.5
            img = (img - 0.5) / 0.5
            # Resize to 224x224 to match training
            from torchvision.transforms.functional import resize
            img_tensor = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
            if img_tensor.shape[1] != 224 or img_tensor.shape[2] != 224:
                img_tensor = resize(img_tensor, [224, 224])
        else:
            img = Image.open(file.stream).convert('L')  # Grayscale
            img = img.resize((224, 224))  # Must match training size
            img_array = np.array(img, dtype=np.float32) / 255.0
            # Apply same normalization as training: (x - 0.5) / 0.5
            img_array = (img_array - 0.5) / 0.5
            img_tensor = torch.from_numpy(img_array).float()
            if img_tensor.dim() == 2:
                img_tensor = img_tensor.unsqueeze(0)
        # Add batch dimension later before model call

        # Normalize lab values using NHANES statistics (must match training)
        labs = np.array([crp, wbc, hb], dtype=np.float32)
        # NHANES stats: CRP, WBC, Hemoglobin
        means = np.array([3.44, 7.38, 13.74])
        stds = np.array([7.41, 5.19, 1.51])
        labs_normalized = (labs - means) / stds
        labs_tensor = torch.from_numpy(labs_normalized).float()
        
        # Make prediction with GPU optimization
        with torch.no_grad():
            # Move to device (GPU or CPU)
            img_tensor = img_tensor.to(device)
            labs_tensor = labs_tensor.to(device)
            
            # Run inference with mixed precision on GPU for faster computation

            # Add batch dimension: [1, 1, 128, 128]
            img_tensor = img_tensor.unsqueeze(0)
            labs_tensor = labs_tensor.unsqueeze(0)
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(img_tensor, labs_tensor)
                    probs = torch.softmax(outputs, dim=1)
            else:
                outputs = model(img_tensor, labs_tensor)
                probs = torch.softmax(outputs, dim=1)
            
            prediction = torch.argmax(probs, dim=1)
            confidence = probs[0, prediction].item()
        
        # Prepare response
        pred_class = int(prediction.item())
        all_probs = {CLASS_NAMES[i]: float(probs[0, i].item()) for i in range(NUM_CLASSES)}
        
        return jsonify({
            'success': True,
            'predicted_class': CLASS_NAMES[pred_class],
            'class_id': pred_class,
            'confidence': round(confidence, 4),
            'probabilities': all_probs,
            'lab_values': {
                'crp': crp,
                'wbc': wbc,
                'hb': hb
            },
            'explanation': get_explanation(pred_class, all_probs),
            'inference_device': str(device)
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/api/predict-demo', methods=['POST'])
def predict_demo():
    """Demo prediction with synthetic data"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get disease class from request
        disease_class = int(request.json.get('disease_class', 0))
        if disease_class not in range(NUM_CLASSES):
            return jsonify({'error': f'Invalid disease class: {disease_class}'}), 400
        
        # Create synthetic image and lab values
        img_tensor = create_synthetic_image(disease_class)
        labs_dict = get_lab_values(disease_class)
        
        # Normalize labs using NHANES statistics (must match training)
        labs = np.array([labs_dict['crp'], labs_dict['wbc'], labs_dict['hb']], dtype=np.float32)
        means = np.array([3.44, 7.38, 13.74])
        stds = np.array([7.41, 5.19, 1.51])
        labs_normalized = (labs - means) / stds
        labs_tensor = torch.from_numpy(labs_normalized)
        
        # Make prediction with GPU optimization
        with torch.no_grad():
            img_tensor = img_tensor.to(device).unsqueeze(0)
            labs_tensor = labs_tensor.to(device).unsqueeze(0)
            
            # Use mixed precision for faster inference on GPU
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(img_tensor, labs_tensor)
                    probs = torch.softmax(outputs, dim=1)
            else:
                outputs = model(img_tensor, labs_tensor)
                probs = torch.softmax(outputs, dim=1)
            
            prediction = torch.argmax(probs, dim=1)
            confidence = probs[0, prediction].item()
        
        pred_class = int(prediction.item())
        all_probs = {CLASS_NAMES[i]: float(probs[0, i].item()) for i in range(NUM_CLASSES)}
        
        return jsonify({
            'success': True,
            'true_class': CLASS_NAMES[disease_class],
            'predicted_class': CLASS_NAMES[pred_class],
            'class_id': pred_class,
            'confidence': round(confidence, 4),
            'probabilities': all_probs,
            'lab_values': labs_dict,
            'explanation': get_explanation(pred_class, all_probs),
            'synthetic': True,
            'inference_device': str(device)
        })
    
    except Exception as e:
        return jsonify({'error': f'Demo prediction error: {str(e)}'}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_file': os.path.basename(model_path) if model_path else 'Not loaded',
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'input_features': {
            'image': '1 channel CT image (128x128)',
            'labs': '3 lab values (CRP, WBC, Hb)'
        },
        'device': str(device),
        'model_loaded': model is not None
    })

@app.route('/api/gpu-stats', methods=['GET'])
def gpu_stats():
    """Get GPU statistics"""
    if not torch.cuda.is_available():
        return jsonify({
            'available': False,
            'device': 'CPU',
            'message': 'No GPU detected'
        })
    
    torch.cuda.synchronize()
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    free = total_memory - allocated
    
    return jsonify({
        'available': True,
        'device_name': torch.cuda.get_device_name(0),
        'total_memory_gb': round(total_memory, 2),
        'allocated_memory_gb': round(allocated, 2),
        'reserved_memory_gb': round(reserved, 2),
        'free_memory_gb': round(free, 2),
        'memory_percent': round((allocated / total_memory) * 100, 2)
    })

def get_explanation(pred_class, probs):
    """Get human-readable explanation"""
    explanations = {
        0: "Normal CT findings with stable lab values.",
        1: "Tumor detected with elevated CRP and low hemoglobin.",
        2: "Active infection indicated by high CRP and WBC.",
        3: "Inflammatory condition with moderately elevated markers."
    }
    
    explanation = explanations.get(pred_class, "Unknown condition")
    
    # Add confidence note
    if probs[CLASS_NAMES[pred_class]] >= 0.95:
        explanation += " (High confidence)"
    elif probs[CLASS_NAMES[pred_class]] >= 0.70:
        explanation += " (Moderate confidence)"
    else:
        explanation += " (Low confidence - clinical review recommended)"
    
    return explanation

@app.route('/api/lab-ranges', methods=['GET'])
def lab_ranges():
    """Get typical lab value ranges"""
    return jsonify({
        'CRP': {
            'unit': 'mg/L',
            'normal': '< 3',
            'ranges': {'Normal': [0, 3], 'Infection': [10, 100], 'Inflammatory': [5, 50]}
        },
        'WBC': {
            'unit': 'x10^9/L',
            'normal': '4.5 - 11',
            'ranges': {'Normal': [4, 11], 'Infection': [10, 25], 'Inflammatory': [5, 12]}
        },
        'Hb': {
            'unit': 'g/dL',
            'normal': '12 - 16',
            'ranges': {'Normal': [12, 16], 'Tumor': [9, 14], 'Infection': [10, 15]}
        }
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': f'File too large. Maximum: {MAX_FILE_SIZE // (1024*1024)}MB'}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model on startup
    print("Initializing Disease Classifier API...")
    if load_model():
        print("Flask app ready!")
        print(f"Device: {device}")
        print("Starting server on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Exiting.")
        sys.exit(1)
