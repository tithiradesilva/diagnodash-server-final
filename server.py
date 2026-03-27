import os
import io
import torch
import torchvision.transforms as transforms
import torchvision.ops as ops
from flask import Flask, request, jsonify
from PIL import Image

# Import exact V2 architecture modules
from model import MobileNetRefineDetLiteCBAM
from utils import AnchorGenerator, decode, xyxy_to_cxcywh

app = Flask(__name__)

# --- CONFIGURATION ---
DEVICE = 'cpu'  # Cloud hosting requires CPU
MODEL_PATH = 'best_model.pth' # Make sure your V2 weight file is named exactly this
IMG_SIZE = 512
CONF_THRESHOLD = 0.30
NMS_THRESHOLD = 0.30

# Exact classes from your V2 training
CLASSES = [
    '__background__', 
    'battery_icon', 
    'engine_icon', 
    'oil_pressure_icon', 
    'parking_brake_icon', 
    'power_steering_icon'
]

# --- GLOBAL INITIALIZATION (Prevents Memory Crashes) ---
print("🚀 Initializing V2 Model & Anchors...")
model = MobileNetRefineDetLiteCBAM(num_classes=6).to(DEVICE)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print("✅ V2 Model Loaded Successfully!")
else:
    print(f"❌ CRITICAL ERROR: {MODEL_PATH} not found.")

anchors = AnchorGenerator(IMG_SIZE).forward(DEVICE)

# Standard ImageNet transform matching your data.py training pipeline
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files['file']
    
    try:
        # 1. Read image natively with PIL (No heavy OpenCV needed)
        img_bytes = file.read()
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w_orig, h_orig = img_pil.size

        # 2. Transform into Tensor
        input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

        # 3. V2 Inference Engine
        with torch.no_grad(): # CRITICAL: Prevents memory leaks
            arm_loc, arm_conf, odm_loc, odm_conf = model(input_tensor)
            
            # Remove batch dimensions
            arm_loc = arm_loc[0]
            odm_loc = odm_loc[0]
            odm_conf = odm_conf[0]

            # V2 RefineDet Decoding Logic
            refined_anchors_xyxy = decode(arm_loc, anchors)
            refined_anchors_cxcywh = xyxy_to_cxcywh(refined_anchors_xyxy)
            boxes = decode(odm_loc, refined_anchors_cxcywh)

            boxes = boxes.clamp(0, 1) # Keep in bounds
            scores = torch.softmax(odm_conf, dim=1)

        # 4. Filter and Find Best Detection
        best_overall_score = 0.0
        best_overall_class = 0
        best_overall_box = None

        # Iterate through classes exactly like your V2 evaluation.py script
        for cls in range(1, len(CLASSES)):
            cls_scores = scores[:, cls]
            mask = cls_scores > CONF_THRESHOLD
            
            if mask.sum() == 0:
                continue

            boxes_cls = boxes[mask]
            scores_cls = cls_scores[mask]

            # Apply Non-Maximum Suppression (NMS)
            keep = ops.nms(boxes_cls, scores_cls, NMS_THRESHOLD)
            
            boxes_cls = boxes_cls[keep]
            scores_cls = scores_cls[keep]

            # Find the absolute best prediction across all classes
            best_idx = scores_cls.argmax()
            current_best_score = scores_cls[best_idx].item()

            if current_best_score > best_overall_score:
                best_overall_score = current_best_score
                best_overall_class = cls
                best_overall_box = boxes_cls[best_idx].tolist()

        # 5. Build Lightweight Payload
        if best_overall_box is not None:
            # Convert normalized [0,1] coordinates back to original pixel dimensions
            x1 = int(best_overall_box[0] * w_orig)
            y1 = int(best_overall_box[1] * h_orig)
            x2 = int(best_overall_box[2] * w_orig)
            y2 = int(best_overall_box[3] * h_orig)

            return jsonify({
                "success": True,
                "detected_class": CLASSES[best_overall_class],
                "confidence": best_overall_score,
                "box": [x1, y1, x2, y2] # Pure math payload
            })
        else:
            return jsonify({
                "success": False,
                "message": "No specific warning light detected."
            })

    except Exception as e:
        print(f"❌ Inference Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)