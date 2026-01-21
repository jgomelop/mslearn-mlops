"""
Inference script for X-ray classification model deployment.
Located in: src/model/score.py
"""

import json
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import base64
import torchxrayvision as xrv


def init():
    """
    This function is called when the container is initialized/started.
    Load your model here.
    """
    global model
    global device
    global class_names
    
    # Define class names (must match training)
    class_names = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia',
        'Mass', 'Nodule', 'Pneumothorax', 'Infiltration'
    ]
    
    # Model configuration (must match training)
    num_classes = len(class_names)
    model_name = "densenet121-res224-all"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get the path to the registered model
    # Azure ML stores the model in AZUREML_MODEL_DIR
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    print(f"Model directory: {model_dir}")
    
    # List contents of model directory to debug
    if model_dir and os.path.exists(model_dir):
        print(f"Contents of model directory: {os.listdir(model_dir)}")
        # The model might be in a subdirectory
        for root, dirs, files in os.walk(model_dir):
            print(f"  {root}: {files}")
    
    # Find the model.pt file
    model_path = None
    if model_dir:
        # Try direct path
        if os.path.exists(os.path.join(model_dir, "model.pt")):
            model_path = os.path.join(model_dir, "model.pt")
        # Try looking in subdirectories
        else:
            for root, dirs, files in os.walk(model_dir):
                if "model.pt" in files:
                    model_path = os.path.join(root, "model.pt")
                    break
    
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model.pt in {model_dir}")
    
    print(f"Loading model from: {model_path}")
    
    # Create model architecture using xrv.models.DenseNet
    print(f"Creating model: {model_name} with {num_classes} classes")
    model = xrv.models.DenseNet(weights=model_name)
    
    # Replace classifier layer
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    
    # Disable pretrained output normalization (CRITICAL)
    model.op_threshs = None
    
    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Move to device and set to eval mode
    model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully")


def preprocess_image(image_data, img_size=224):
    """
    Preprocess the input image for model inference.
    Follows the same preprocessing as training.
    
    Args:
        image_data: Either a base64 encoded string or raw bytes
        img_size: Target image size for resizing
    
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    try:
        # Decode base64 if needed
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
        
        # Open image and convert to grayscale
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = np.array(img)
        
        # Center crop to square (same as training)
        h, w = img.shape
        m = min(h, w)
        img = img[(h - m) // 2: (h - m) // 2 + m, (w - m) // 2: (w - m) // 2 + m]
        
        # Resize using torchxrayvision resizer
        transform = xrv.datasets.XRayResizer(img_size)
        img = transform(img)
        
        # Remove extra dimensions if present
        if img.ndim == 3:
            img = img.squeeze()
        
        # Normalize (same as training)
        img = xrv.datasets.normalize(img, 255)
        
        # Add batch and channel dimensions: (1, 1, H, W)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        
        return img_tensor
        
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint.
    
    Args:
        raw_data: The raw request data as a string
    
    Returns:
        Predictions as a JSON string
    """
    try:
        # Parse input data
        data = json.loads(raw_data)
        
        # Get image data
        if "image" in data:
            image_data = data["image"]
        elif "data" in data:
            image_data = data["data"]
        else:
            return json.dumps({
                "error": "No image data provided. Expected 'image' or 'data' field in JSON."
            })
        
        # Get image size parameter (default to 224 as in training)
        img_size = data.get("img_size", 224)
        
        # Preprocess image
        img_tensor = preprocess_image(image_data, img_size)
        img_tensor = img_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(img_tensor)
            
            # Apply sigmoid for multi-label classification
            probabilities = torch.sigmoid(output)
            probabilities = probabilities.cpu().numpy()[0]
        
        # Create predictions dictionary
        predictions = {}
        for i, class_name in enumerate(class_names):
            predictions[class_name] = float(probabilities[i])
        
        # Sort by probability (descending)
        sorted_predictions = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))
        
        # Create result
        result = {
            "predictions": sorted_predictions,
            "top_prediction": max(predictions, key=predictions.get),
            "top_probability": float(max(probabilities)),
            "model_info": {
                "num_classes": len(class_names),
                "classes": class_names
            }
        }
        
        return json.dumps(result)
        
    except Exception as e:
        error_msg = f"Error during inference: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return json.dumps({"error": error_msg})